# -*- coding: utf-8 -*-
"""
Modules de Détection pour le Système Drone
Contient: YOLO+DeepSORT, MediaPipe, Motion Tracking, et Fusion
"""

import time
from collections import defaultdict, deque

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print(" DeepSORT non disponible - utilisation du tracking OpenCV")


class BaseDetector:
    """Classe de base pour tous les détecteurs"""
    
    def __init__(self):
        self.trajectories = defaultdict(lambda: deque(maxlen=30))
        self.person_stats = defaultdict(lambda: {
            'first_seen': time.time(),
            'last_seen': time.time(),
            'positions': deque(maxlen=100),
            'speeds': deque(maxlen=20),
            'danger_level': 0
        })
        self.colors = self._generate_colors(50)
    
    def _generate_colors(self, n):
        """Génère n couleurs distinctes"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def calculate_speed(self, person_id, current_pos, fps=30):
        """Calcule la vitesse de déplacement"""
        stats = self.person_stats[person_id]
        stats['positions'].append(current_pos)
        
        if len(stats['positions']) < 2:
            return 0
        
        prev_pos = stats['positions'][-2]
        distance = np.sqrt(
            (current_pos[0] - prev_pos[0])**2 + 
            (current_pos[1] - prev_pos[1])**2
        )
        
        speed = distance * fps
        stats['speeds'].append(speed)
        return speed
    
    def detect(self, frame, **kwargs):
        """Méthode abstraite - à implémenter par les sous-classes"""
        raise NotImplementedError


class YOLODeepSORTDetector(BaseDetector):
    """Détecteur YOLO + DeepSORT"""
    
    def __init__(self, model='yolov8n.pt', confidence_threshold=0.5):
        super().__init__()
        
        print(" Chargement YOLO...")
        self.yolo = YOLO(model)
        self.confidence_threshold = confidence_threshold
        
        if DEEPSORT_AVAILABLE:
            print(" Initialisation DeepSORT...")
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                embedder="mobilenet",
                half=True,
                embedder_gpu=False
            )
            self.use_deepsort = True
        else:
            print(" Utilisation du tracker OpenCV")
            self.tracker = cv2.legacy.MultiTracker_create()
            self.use_deepsort = False
            self.next_id = 1
    
    def detect(self, frame, show_skeleton=False, show_trajectory=True):
        """Détecte les personnes avec YOLO et les track"""
        
        # Détection YOLO
        results = self.yolo(frame, classes=[0], verbose=False)[0]
        
        detections = []
        bboxes = []
        
        for box in results.boxes:
            if box.conf[0] > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if self.use_deepsort:
                    detections.append(([x1, y1, x2-x1, y2-y1], confidence, 'person'))
                else:
                    bboxes.append((x1, y1, x2, y2, confidence))
        
        # Tracking
        tracked_persons = []
        
        if self.use_deepsort and detections:
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                tracked_persons.append({
                    'id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'confidence': 1.0
                })
        
        elif bboxes:
            # Tracking simple avec OpenCV
            for x1, y1, x2, y2, conf in bboxes:
                tracked_persons.append({
                    'id': self.next_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'confidence': conf
                })
                self.next_id += 1
        
        # Visualisation
        vis_frame = self.draw_detections(
            frame.copy(), 
            tracked_persons, 
            show_trajectory
        )
        
        return {
            'frame': vis_frame,
            'detections': tracked_persons,
            'method': 'YOLO+DeepSORT' if self.use_deepsort else 'YOLO+OpenCV'
        }
    
    def draw_detections(self, frame, persons, show_trajectory):
        """Dessine les détections sur l'image"""
        
        for person in persons:
            person_id = person['id']
            x1, y1, x2, y2 = person['bbox']
            center = person['center']
            
            color = self.colors[int(person_id) % len(self.colors)]
            
            # Boîte
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ID
            label = f"ID:{person_id} ({person['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Trajectoire
            if show_trajectory:
                self.trajectories[person_id].append(center)
                points = list(self.trajectories[person_id])
                
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], color, 2)
            
            # Vitesse
            speed = self.calculate_speed(person_id, center)
            cv2.putText(frame, f"{speed:.1f} px/s",
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame


class MediaPipePoseDetector(BaseDetector):
    """Détecteur MediaPipe Pose"""
    
    def __init__(self):
        super().__init__()
        
        print(" Initialisation MediaPipe...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.next_id = 1
    
    def detect(self, frame, show_skeleton=True, show_trajectory=True):
        """Détecte les poses avec MediaPipe"""
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        vis_frame = frame.copy()
        detections = []
        
        if results.pose_landmarks:
            # Dessiner le squelette
            if show_skeleton:
                self.mp_drawing.draw_landmarks(
                    vis_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Calculer la bounding box
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            
            # Analyse de pose
            danger_level, pose_type = self.analyze_pose(landmarks)
            
            person = {
                'id': self.next_id,
                'bbox': (x1, y1, x2, y2),
                'center': ((x1+x2)//2, (y1+y2)//2),
                'confidence': 1.0,
                'pose_type': pose_type,
                'danger_level': danger_level
            }
            
            detections.append(person)
            
            # Annotation
            color = (0, 255, 0) if danger_level == 0 else (0, 165, 255) if danger_level == 1 else (0, 0, 255)
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"Pose: {pose_type}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if danger_level > 0:
                cv2.putText(vis_frame, "⚠️ ALERTE",
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return {
            'frame': vis_frame,
            'detections': detections,
            'method': 'MediaPipe'
        }
    
    def analyze_pose(self, landmarks):
        """Analyse la pose pour détecter des situations anormales"""
        
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        
        # Personne allongée
        if abs(avg_shoulder_y - avg_hip_y) < 0.15 and nose.y > avg_shoulder_y:
            return 3, "ALLONGÉ"
        
        # Bras levés (détresse)
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        if left_wrist.y < left_shoulder.y - 0.2 and right_wrist.y < right_shoulder.y - 0.2:
            return 2, "DÉTRESSE"
        
        # Position normale
        return 0, "NORMAL"


class MotionTracker(BaseDetector):
    """Détecteur basé sur le mouvement"""
    
    def __init__(self):
        super().__init__()
        
        print(" Initialisation Motion Tracker...")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        self.next_id = 1
        self.last_positions = {}
    
    def detect(self, frame, show_skeleton=False, show_trajectory=True):
        """Détecte le mouvement dans l'image"""
        
        # Soustraction de fond
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Nettoyage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vis_frame = frame.copy()
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrer les petits mouvements
            if area < 2000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Vérifier les proportions (humain)
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 1.2 or aspect_ratio > 4:
                continue
            
            center = (x + w//2, y + h//2)
            
            # Trouver ID le plus proche
            person_id = self.match_or_create_id(center)
            
            person = {
                'id': person_id,
                'bbox': (x, y, x+w, y+h),
                'center': center,
                'confidence': min(area / 10000, 1.0),
                'motion_area': area
            }
            
            detections.append(person)
            
            # Visualisation
            color = self.colors[person_id % len(self.colors)]
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis_frame, f"Motion ID:{person_id}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Trajectoire
            if show_trajectory:
                self.trajectories[person_id].append(center)
                points = list(self.trajectories[person_id])
                
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)
        
        return {
            'frame': vis_frame,
            'detections': detections,
            'method': 'Motion'
        }
    
    def match_or_create_id(self, position, max_distance=100):
        """Trouve l'ID le plus proche ou en crée un nouveau"""
        
        min_dist = float('inf')
        best_id = None
        
        for person_id, last_pos in self.last_positions.items():
            dist = np.linalg.norm(np.array(position) - np.array(last_pos))
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_id = person_id
        
        if best_id is None:
            best_id = self.next_id
            self.next_id += 1
        
        self.last_positions[best_id] = position
        return best_id


class FusionDetector:
    """Détecteur fusionnant les 3 approches"""
    
    def __init__(self, yolo_detector, mediapipe_detector, motion_detector):
        self.yolo_detector = yolo_detector
        self.mediapipe_detector = mediapipe_detector
        self.motion_detector = motion_detector
        
        print(" Détecteur fusionné créé")
    
    def detect(self, frame, show_skeleton=True, show_trajectory=True):
        """Fusionne les résultats des 3 détecteurs"""
        
        # Exécuter chaque détecteur
        yolo_result = self.yolo_detector.detect(frame, show_trajectory=False)
        mediapipe_result = self.mediapipe_detector.detect(frame, show_skeleton=False)
        motion_result = self.motion_detector.detect(frame, show_trajectory=False)
        
        # Fusionner les détections
        all_detections = []
        all_detections.extend(yolo_result['detections'])
        all_detections.extend(mediapipe_result['detections'])
        all_detections.extend(motion_result['detections'])
        
        # Supprimer les doublons (détections qui se chevauchent)
        unique_detections = self.remove_duplicates(all_detections)
        
        # Créer la frame de visualisation
        vis_frame = frame.copy()
        
        for person in unique_detections:
            x1, y1, x2, y2 = person['bbox']
            center = person['center']
            
            # Couleur selon le niveau de danger
            danger = person.get('danger_level', 0)
            if danger >= 3:
                color = (0, 0, 255)  # Rouge
            elif danger >= 2:
                color = (0, 165, 255)  # Orange
            elif danger >= 1:
                color = (0, 255, 255)  # Jaune
            else:
                color = (0, 255, 0)  # Vert
            
            # Boîte
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Label
            sources = person.get('sources', [])
            label = f"ID:{person['id']} [{','.join(sources)}]"
            cv2.putText(vis_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Info supplémentaire
            if 'pose_type' in person:
                cv2.putText(vis_frame, person['pose_type'],
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Stats dans le coin
        cv2.putText(vis_frame, f"FUSION: {len(unique_detections)} personnes",
                   (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"YOLO: {len(yolo_result['detections'])}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(vis_frame, f"MediaPipe: {len(mediapipe_result['detections'])}",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(vis_frame, f"Motion: {len(motion_result['detections'])}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return {
            'frame': vis_frame,
            'detections': unique_detections,
            'method': 'FUSION',
            'details': {
                'yolo': len(yolo_result['detections']),
                'mediapipe': len(mediapipe_result['detections']),
                'motion': len(motion_result['detections'])
            }
        }
    
    def remove_duplicates(self, detections, iou_threshold=0.5):
        """Supprime les détections en double basées sur l'IoU"""
        
        if not detections:
            return []
        
        # Calculer l'IoU entre toutes les paires
        unique = []
        
        for det in detections:
            is_duplicate = False
            
            for unique_det in unique:
                iou = self.calculate_iou(det['bbox'], unique_det['bbox'])
                
                if iou > iou_threshold:
                    # C'est un doublon - fusionner les infos
                    unique_det.setdefault('sources', []).append(det.get('method', 'unknown'))
                    
                    # Garder le danger level le plus élevé
                    if det.get('danger_level', 0) > unique_det.get('danger_level', 0):
                        unique_det['danger_level'] = det['danger_level']
                        unique_det['pose_type'] = det.get('pose_type', 'UNKNOWN')
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                det['sources'] = [det.get('method', 'unknown')]
                unique.append(det)
        
        return unique
    
    def calculate_iou(self, bbox1, bbox2):
        """Calcule l'Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0