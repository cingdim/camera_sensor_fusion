"""LightGlue-based fallback for ArUco marker detection.

When OpenCV ArUco fails to detect expected markers, this module uses
SuperPoint+LightGlue feature matching with homography-based corner recovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TrackerState:
    """Tracks last known state of a marker for temporal coherence."""
    marker_id: int
    last_corners: np.ndarray  # (4, 2) float32
    last_seen_frame_index: int
    last_frame_gray: Optional[np.ndarray] = None


@dataclass
class DebugInfo:
    """Debug information for recovered markers."""
    marker_id: int
    source: str  # "aruco", "lightglue_reacquire", "lightglue_track"
    inliers: int
    match_quality: float
    homography: Optional[np.ndarray] = None


class LightGlueFallback:
    """LightGlue-based fallback system for recovering missing ArUco markers.
    
    Uses SuperPoint feature detection and LightGlue matching to:
    1. Track markers across frames using optical flow
    2. Re-acquire markers by matching against templates
    3. Compute homographies to recover corner positions
    """

    def __init__(self, lightglue_cfg, aruco_cfg):
        """Initialize LightGlue fallback system.
        
        Args:
            lightglue_cfg: LightGlueConfig with settings
            aruco_cfg: ArUco configuration for marker info
        """
        self.cfg = lightglue_cfg
        self.aruco_cfg = aruco_cfg
        self.device = lightglue_cfg.device
        self.enabled = lightglue_cfg.enabled
        
        # Tracker state per marker ID
        self.tracker_states: Dict[int, TrackerState] = {}
        self.current_frame_index = 0
        
        # Reacquire interval tracking: marker_id -> last_reacquire_frame_index
        self.last_reacquire_frame: Dict[int, int] = {}
        
        # ArUco detector for ID verification
        self.aruco_detector = None
        
        # Template data: marker_id -> dict with image, keypoints (torch), descriptors (torch), corners
        self.templates: Dict[int, Dict] = {}
        
        # SuperPoint + LightGlue models
        self.superpoint = None
        self.lightglue = None
        self.torch_available = False
        
        if not self.enabled:
            logger.info("LightGlue fallback is disabled")
            return
        
        # Try to import and initialize torch/lightglue
        try:
            import torch
            self.torch_available = True
            logger.info(f"PyTorch available, using device: {self.device}")
            
            # Import SuperPoint and LightGlue
            try:
                from lightglue import SuperPoint, LightGlue
                from lightglue.utils import load_image, rbd
                
                self.SuperPoint = SuperPoint
                self.LightGlue = LightGlue
                self.load_image = load_image
                self.rbd = rbd
                
                # Initialize models
                self.superpoint = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
                self.lightglue = LightGlue(features='superpoint').eval().to(self.device)
                
                logger.info("SuperPoint and LightGlue models initialized successfully")
                
                # Initialize ArUco detector for ID verification
                if self.cfg.verify_id:
                    self._init_aruco_detector()
                    
            except ImportError as e:
                logger.warning(f"LightGlue package not available: {e}")
                logger.warning("Install with: pip install lightglue")
                self.enabled = False
                return
                
        except ImportError:
            logger.warning("PyTorch not available - LightGlue fallback disabled")
            logger.warning("Install with: pip install torch")
            self.enabled = False
            return
        
        # Load templates
        self._load_templates()
    
    def _init_aruco_detector(self):
        """Initialize ArUco detector for ID verification."""
        try:
            from camera_fusion.detect import build_detector
            _, _, self.aruco_detector = build_detector(self.aruco_cfg)
            logger.info("ArUco detector initialized for ID verification")
        except Exception as e:
            logger.warning(f"Failed to initialize ArUco detector for verification: {e}")
            self.aruco_detector = None
    
    def _verify_marker_id(
        self,
        frame_gray: np.ndarray,
        corners: np.ndarray,
        expected_id: int
    ) -> bool:
        """Verify that the warped patch at corners contains the expected ArUco ID.
        
        Args:
            frame_gray: Frame in grayscale
            corners: (4, 2) corner coordinates
            expected_id: Expected marker ID
        
        Returns:
            True if ID matches, False otherwise
        """
        if not self.cfg.verify_id or self.aruco_detector is None:
            return True  # Skip verification if disabled or detector unavailable
        
        try:
            # Define destination square (200x200 px)
            patch_size = 200
            dst_corners = np.array([
                [0, 0],
                [patch_size - 1, 0],
                [patch_size - 1, patch_size - 1],
                [0, patch_size - 1]
            ], dtype=np.float32)
            
            # Compute homography to warp quadrilateral to square
            H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
            
            # Warp the patch
            warped = cv2.warpPerspective(frame_gray, H, (patch_size, patch_size))
            
            # Run ArUco detection on warped patch
            if self.aruco_detector is not None:
                corners_detected, ids, _ = self.aruco_detector.detectMarkers(warped)
            else:
                # Fallback to legacy API
                from camera_fusion.detect import build_detector
                dictionary, params, _ = build_detector(self.aruco_cfg)
                corners_detected, ids, _ = cv2.aruco.detectMarkers(warped, dictionary, parameters=params)
            
            # Check if expected ID was detected
            if ids is not None and len(ids) > 0:
                detected_ids = ids.flatten()
                if expected_id in detected_ids:
                    logger.debug(f"ID verification passed for marker {expected_id}")
                    return True
                else:
                    logger.warning(f"ID verification failed: expected {expected_id}, got {detected_ids}")
                    return False
            else:
                logger.warning(f"ID verification failed: no marker detected in warped patch for ID {expected_id}")
                return False
                
        except Exception as e:
            logger.debug(f"ID verification failed with exception for marker {expected_id}: {e}")
            return False
    
    def _load_templates(self):
        """Load marker templates and precompute their features.
        
        Template features (keypoints & descriptors) are computed ONCE and cached
        as torch tensors on the configured device. This avoids recomputing them
        per frame during reacquire attempts.
        """
        if not self.enabled:
            return
        
        template_dir = Path(self.cfg.template_dir)
        if not template_dir.exists():
            logger.warning(f"Template directory not found: {template_dir}")
            logger.warning("LightGlue fallback will not be able to recover markers")
            return
        
        import torch
        
        # Look for templates named id_<ID>.png
        for template_path in template_dir.glob("id_*.png"):
            try:
                marker_id = int(template_path.stem.split('_')[1])
            except (IndexError, ValueError):
                logger.warning(f"Invalid template filename: {template_path.name}")
                continue
            
            # Load template image
            img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Failed to load template: {template_path}")
                continue
            
            # Extract features using SuperPoint (ONCE at init time)
            h, w = img.shape
            
            # Convert to torch tensor and normalize
            img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).to(self.device)[None, None]
            
            with torch.no_grad():
                feats = self.superpoint({'image': img_tensor})
            
            # Keep keypoints and descriptors as torch tensors on device
            # feats['keypoints'] is shape (1, N, 2)
            # feats['descriptors'] is shape (1, 256, N)
            template_kpts = feats['keypoints'][0]  # (N, 2) on device
            template_desc = feats['descriptors'][0].T  # (N, 256) on device
            
            # Define template corners (assuming template is the marker square)
            corners = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1]
            ], dtype=np.float32)
            
            # Store template data with torch tensors
            self.templates[marker_id] = {
                'image': img,
                'keypoints': template_kpts,  # torch.Tensor (N, 2) on device
                'descriptors': template_desc,  # torch.Tensor (N, 256) on device
                'corners': corners
            }
            logger.info(f"Loaded template for marker {marker_id}: {template_kpts.shape[0]} keypoints (cached as torch tensors)")
        
        logger.info(f"Loaded {len(self.templates)} marker templates with cached torch features")
    
    def recover_missing(
        self,
        frame_bgr: np.ndarray,
        detected_dict: Dict[int, np.ndarray],
        expected_ids: Set[int]
    ) -> Tuple[Dict[int, np.ndarray], List[DebugInfo]]:
        """Recover missing markers using LightGlue.
        
        Args:
            frame_bgr: Current frame in BGR format
            detected_dict: Dict mapping marker_id -> corners (4, 2) for detected markers
            expected_ids: Set of marker IDs we expect to see
        
        Returns:
            (updated_detected_dict, debug_info_list)
        """
        if not self.enabled:
            return detected_dict, []
        
        debug_info: List[DebugInfo] = []
        updated_dict = detected_dict.copy()
        
        # Increment frame counter
        self.current_frame_index += 1
        
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Update tracker states for detected markers (with proper source annotation)
        for marker_id, corners in detected_dict.items():
            self.tracker_states[marker_id] = TrackerState(
                marker_id=marker_id,
                last_corners=corners.copy(),
                last_seen_frame_index=self.current_frame_index,
                last_frame_gray=frame_gray.copy()
            )
            debug_info.append(DebugInfo(
                marker_id=marker_id,
                source="aruco",
                inliers=0,
                match_quality=1.0
            ))
        
        # Identify missing markers
        detected_ids = set(detected_dict.keys())
        missing_ids = expected_ids - detected_ids
        
        if not missing_ids:
            return updated_dict, debug_info
        
        logger.debug(f"Frame {self.current_frame_index}: missing markers {missing_ids}")
        
        # SAFETY GUARDRAIL: Limit number of fallback attempts per frame
        max_attempts = self.cfg.max_fallback_markers_per_frame
        missing_ids_list = list(missing_ids)
        
        # Prioritize markers by recency (most recently seen first)
        missing_ids_sorted = sorted(
            missing_ids_list,
            key=lambda mid: self.tracker_states.get(mid, TrackerState(mid, None, -999)).last_seen_frame_index,
            reverse=True
        )
        
        recovery_attempts = 0
        
        # Attempt recovery for each missing marker (up to max_attempts)
        for marker_id in missing_ids_sorted:
            if recovery_attempts >= max_attempts:
                logger.debug(f"Reached max fallback attempts ({max_attempts}) for this frame")
                break
            
            recovery_attempts += 1
            recovered_corners, info = self._recover_marker(
                marker_id, frame_gray, frame_bgr
            )
            if recovered_corners is not None:
                # Verify corners are in correct OpenCV ArUco format: (4, 2) float32
                if not isinstance(recovered_corners, np.ndarray):
                    recovered_corners = np.array(recovered_corners, dtype=np.float32)
                if recovered_corners.shape != (4, 2):
                    logger.warning(f"Invalid corner shape for marker {marker_id}: {recovered_corners.shape}")
                    continue
                if recovered_corners.dtype != np.float32:
                    recovered_corners = recovered_corners.astype(np.float32)
                
                updated_dict[marker_id] = recovered_corners
                debug_info.append(info)
                logger.info(f"Recovered marker {marker_id} via {info.source}")
        
        return updated_dict, debug_info
    
    def _recover_marker(
        self,
        marker_id: int,
        frame_gray: np.ndarray,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[DebugInfo]]:
        """Attempt to recover a single missing marker.
        
        Strategy:
        1. If marker was recently seen, try tracking
        2. If tracking fails or marker not recent, check reacquire interval
        3. Try template reacquire if interval allows
        
        Args:
            marker_id: ID of marker to recover
            frame_gray: Current frame in grayscale
            frame_bgr: Current frame in BGR
        
        Returns:
            (corners, debug_info) or (None, None) if recovery failed
        """
        # Try tracking first if marker was recently seen
        tracker_state = self.tracker_states.get(marker_id)
        if tracker_state is not None:
            age = self.current_frame_index - tracker_state.last_seen_frame_index
            if age <= self.cfg.max_age_frames and tracker_state.last_frame_gray is not None:
                corners, info = self._track_marker(
                    marker_id, tracker_state, frame_gray
                )
                if corners is not None:
                    # Verify ID
                    if self._verify_marker_id(frame_gray, corners, marker_id):
                        return corners, info
                    else:
                        logger.debug(f"Tracking result failed ID verification for marker {marker_id}")
        
        # SAFETY GUARDRAIL: Check reacquire interval
        last_reacquire = self.last_reacquire_frame.get(marker_id, -999)
        frames_since_reacquire = self.current_frame_index - last_reacquire
        
        if frames_since_reacquire < self.cfg.reacquire_interval_frames:
            logger.debug(
                f"Skipping reacquire for marker {marker_id}: only {frames_since_reacquire} frames "
                f"since last attempt (min interval: {self.cfg.reacquire_interval_frames})"
            )
            return None, None
        
        # Fall back to template reacquire
        corners, info = self._reacquire_from_template(marker_id, frame_gray, frame_bgr)
        
        if corners is not None:
            # Update last reacquire frame
            self.last_reacquire_frame[marker_id] = self.current_frame_index
            
            # Verify ID
            if self._verify_marker_id(frame_gray, corners, marker_id):
                return corners, info
            else:
                logger.warning(f"Template reacquire result failed ID verification for marker {marker_id}")
                return None, None
        
        return None, None
    
    def _track_marker(
        self,
        marker_id: int,
        tracker_state: TrackerState,
        frame_gray: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[DebugInfo]]:
        """Track marker from last known position using optical flow in ROI.
        
        Args:
            marker_id: ID of marker to track
            tracker_state: Last known state
            frame_gray: Current frame
        
        Returns:
            (corners, debug_info) or (None, None) if tracking failed
        """
        try:
            # Define ROI around last known position
            last_corners = tracker_state.last_corners
            x_min = int(np.floor(last_corners[:, 0].min())) - self.cfg.roi_expand_px
            x_max = int(np.ceil(last_corners[:, 0].max())) + self.cfg.roi_expand_px
            y_min = int(np.floor(last_corners[:, 1].min())) - self.cfg.roi_expand_px
            y_max = int(np.ceil(last_corners[:, 1].max())) + self.cfg.roi_expand_px
            
            h, w = frame_gray.shape
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            if x_max <= x_min or y_max <= y_min:
                return None, None
            
            # Extract ROIs
            prev_roi = tracker_state.last_frame_gray[y_min:y_max, x_min:x_max]
            curr_roi = frame_gray[y_min:y_max, x_min:x_max]
            
            # Adjust corners to ROI coordinates
            roi_corners = last_corners.copy()
            roi_corners[:, 0] -= x_min
            roi_corners[:, 1] -= y_min
            
            # Use optical flow to track corners
            next_corners, status, err = cv2.calcOpticalFlowPyrLK(
                prev_roi,
                curr_roi,
                roi_corners.astype(np.float32),
                None,
                winSize=(21, 21),
                maxLevel=3
            )
            
            # Check if all corners were successfully tracked
            if next_corners is None or status is None or not np.all(status):
                return None, None
            
            # Convert back to image coordinates
            tracked_corners = next_corners.copy()
            tracked_corners[:, 0] += x_min
            tracked_corners[:, 1] += y_min
            
            # Sanity check: ensure corners are within frame
            if (tracked_corners < 0).any() or \
               (tracked_corners[:, 0] >= w).any() or \
               (tracked_corners[:, 1] >= h).any():
                return None, None
            
            # Refine corners if enabled
            if self.cfg.corner_refine:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                tracked_corners = cv2.cornerSubPix(
                    frame_gray,
                    tracked_corners,
                    (5, 5),
                    (-1, -1),
                    criteria
                )
            
            info = DebugInfo(
                marker_id=marker_id,
                source="lg_track",
                inliers=4,
                match_quality=1.0 - err.mean() if err is not None else 1.0
            )
            
            return tracked_corners.reshape(4, 2).astype(np.float32), info
            
        except Exception as e:
            logger.debug(f"Tracking failed for marker {marker_id}: {e}")
            return None, None
    
    def _reacquire_from_template(
        self,
        marker_id: int,
        frame_gray: np.ndarray,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[DebugInfo]]:
        """Re-acquire marker by matching template using LightGlue.
        
        Uses cached template features (computed once at init) to avoid recomputation.
        Prefers ROI-based matching if last_corners exist, otherwise full frame.
        
        Args:
            marker_id: ID of marker to recover
            frame_gray: Current frame in grayscale
            frame_bgr: Current frame in BGR
        
        Returns:
            (corners, debug_info) or (None, None) if reacquire failed
        """
        if marker_id not in self.templates:
            logger.debug(f"No template available for marker {marker_id}")
            return None, None
        
        try:
            import torch
            
            template_data = self.templates[marker_id]
            template_kpts = template_data['keypoints']  # Cached torch tensor (N, 2)
            template_desc = template_data['descriptors']  # Cached torch tensor (N, 256)
            template_corners = template_data['corners']
            
            # SAFETY GUARDRAIL: Prefer ROI-based matching if last_corners exist
            search_region = frame_gray
            roi_offset_x = 0
            roi_offset_y = 0
            
            if self.cfg.prefer_roi_matching and marker_id in self.tracker_states:
                tracker_state = self.tracker_states[marker_id]
                if tracker_state.last_corners is not None:
                    # Define search ROI around last known position (larger expansion)
                    roi_expand = self.cfg.roi_expand_px * 3  # 3x expansion for reacquire
                    last_corners = tracker_state.last_corners
                    
                    x_min = int(np.floor(last_corners[:, 0].min())) - roi_expand
                    x_max = int(np.ceil(last_corners[:, 0].max())) + roi_expand
                    y_min = int(np.floor(last_corners[:, 1].min())) - roi_expand
                    y_max = int(np.ceil(last_corners[:, 1].max())) + roi_expand
                    
                    h_full, w_full = frame_gray.shape
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w_full, x_max)
                    y_max = min(h_full, y_max)
                    
                    if x_max > x_min and y_max > y_min:
                        search_region = frame_gray[y_min:y_max, x_min:x_max]
                        roi_offset_x = x_min
                        roi_offset_y = y_min
                        logger.debug(f"Using ROI search for marker {marker_id}: ({x_min},{y_min})-({x_max},{y_max})")
            
            # Extract features from search region
            h, w = search_region.shape
            frame_tensor = torch.from_numpy(search_region.astype(np.float32) / 255.0).to(self.device)[None, None]
            
            with torch.no_grad():
                frame_feats = self.superpoint({'image': frame_tensor})
                frame_kpts = frame_feats['keypoints'][0]  # (N, 2) on device
                frame_desc = frame_feats['descriptors'][0].T  # (N, 256) on device
                
                # Call LightGlue with flattened feature dicts using rbd()
                # Prepare features dict for LightGlue
                feats0 = {'keypoints': template_kpts, 'descriptors': template_desc}
                feats1 = {'keypoints': frame_kpts, 'descriptors': frame_desc}
                
                # Match features using LightGlue with flattened dict
                matches = self.lightglue({
                    **self.rbd(feats0),
                    **self.rbd(feats1)
                })
            
            # Extract matched keypoints using matches0
            matches0 = matches['matches0']  # Shape (num_keypoints_template,)
            valid = matches0 >= 0  # Boolean mask for valid matches
            
            if valid.sum() < self.cfg.min_inliers:
                logger.debug(f"Insufficient matches for marker {marker_id}: {valid.sum()}")
                return None, None
            
            # Get ONLY matched keypoints (not all keypoints)
            template_matched_idx = torch.where(valid)[0]  # Indices of matched template keypoints
            frame_matched_idx = matches0[valid]  # Indices of matched frame keypoints
            
            # Extract matched point coordinates (convert to numpy for homography)
            template_pts = template_kpts[template_matched_idx].cpu().numpy()  # (M, 2)
            frame_pts = frame_kpts[frame_matched_idx].cpu().numpy()  # (M, 2)
            
            # Compute homography using ONLY matched keypoints
            H, mask = cv2.findHomography(
                template_pts,
                frame_pts,
                cv2.RANSAC,
                5.0
            )
            
            if H is None:
                logger.debug(f"Homography computation failed for marker {marker_id}")
                return None, None
            
            inliers = mask.sum() if mask is not None else 0
            if inliers < self.cfg.min_inliers:
                logger.debug(f"Insufficient inliers for marker {marker_id}: {inliers}")
                return None, None
            
            # Project template corners to frame (within search region)
            corners_transformed = cv2.perspectiveTransform(
                template_corners.reshape(1, 4, 2),
                H
            )
            recovered_corners = corners_transformed.reshape(4, 2).astype(np.float32)
            
            # Adjust corners back to full frame coordinates if ROI was used
            if roi_offset_x != 0 or roi_offset_y != 0:
                recovered_corners[:, 0] += roi_offset_x
                recovered_corners[:, 1] += roi_offset_y
            
            # Sanity check: ensure corners are within full frame
            h_full, w_full = frame_gray.shape
            if (recovered_corners < 0).any() or \
               (recovered_corners[:, 0] >= w_full).any() or \
               (recovered_corners[:, 1] >= h_full).any():
                logger.debug(f"Recovered corners out of bounds for marker {marker_id}")
                return None, None
            
            # Refine corners if enabled
            if self.cfg.corner_refine:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                recovered_corners = cv2.cornerSubPix(
                    frame_gray,
                    recovered_corners,
                    (5, 5),
                    (-1, -1),
                    criteria
                )
            
            match_quality = inliers / valid.sum() if valid.sum() > 0 else 0.0
            
            info = DebugInfo(
                marker_id=marker_id,
                source="lg_reacquire",
                inliers=int(inliers),
                match_quality=match_quality,
                homography=H
            )
            
            return recovered_corners, info
            
        except Exception as e:
            logger.debug(f"Template reacquire failed for marker {marker_id}: {e}")
            return None, None
