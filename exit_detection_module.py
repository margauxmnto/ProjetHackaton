# -*- coding: utf-8 -*-
"""
Module de Détection et Gestion des Sorties de Secours
Pour système de drone de recherche et sauvetage
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ExitDetector:
    """
    Classe pour détecter et gérer les sorties de secours
    - Détection automatique (panneaux verts)
    - Définition manuelle
    - Calcul de distance et direction
    - Sauvegarde/chargement
    """
    
    def __init__(self):
        self.exit_signs = []  # Liste des sorties détectées
        self.manual_exits = []  # Sorties définies manuellement
        self.last_detection_time = 0
        self.detection_interval = 2  # Détecte toutes les 2 secondes
        self.auto_detection_enabled = False
        
        # Paramètres de détection
        self.min_area = 1000
        self.max_area = 30000
        self.min_aspect_ratio = 1.8
        self.max_aspect_ratio = 3.5
        
        # Mode définition
        self.definition_mode = False
        self.temp_exit_position = None
        
        # Historique
        self.detection_history = []
    
    def toggle_auto_detection(self):
        """Active/désactive la détection automatique"""
        self.auto_detection_enabled = not self.auto_detection_enabled
        status = "ACTIVÉE" if self.auto_detection_enabled else "DÉSACTIVÉE"
        print(f"🚨 Détection automatique des sorties: {status}")
        return self.auto_detection_enabled
    
    def add_manual_exit(self, position, label="SORTIE", exit_type="manual"):
        """
        Ajoute manuellement une sortie de secours
        
        Args:
            position: (x, y) coordonnées du centre
            label: Nom de la sortie
            exit_type: Type (manual, entrance, emergency, etc.)
        """
        exit_data = {
            'center': position,
            'label': label,
            'type': exit_type,
            'bbox': (position[0]-50, position[1]-30, 100, 60),
            'timestamp': datetime.now().isoformat(),
            'source': 'manual'
        }
        
        self.manual_exits.append(exit_data)
        print(f"✅ Sortie '{label}' ajoutée à {position}")
        
        # Ajouter à l'historique
        self.detection_history.append({
            'action': 'add',
            'exit': exit_data,
            'timestamp': datetime.now().isoformat()
        })
        
        return exit_data
    
    def remove_exit(self, position, tolerance=50):
        """
        Supprime une sortie proche de la position donnée
        
        Args:
            position: (x, y) position du clic
            tolerance: distance maximale en pixels
        """
        removed = []
        
        # Chercher dans les sorties manuelles
        for i, exit_sign in enumerate(self.manual_exits):
            center = exit_sign['center']
            distance = np.sqrt((center[0] - position[0])**2 + (center[1] - position[1])**2)
            
            if distance < tolerance:
                removed_exit = self.manual_exits.pop(i)
                removed.append(removed_exit)
                print(f"🗑️ Sortie '{removed_exit['label']}' supprimée")
                break
        
        return removed
    
    def remove_last_exit(self):
        """Supprime la dernière sortie ajoutée"""
        if self.manual_exits:
            removed = self.manual_exits.pop()
            print(f" Dernière sortie supprimée: {removed['label']}")
            return removed
        return None
    
    def clear_exits(self):
        """Efface toutes les sorties manuelles"""
        count = len(self.manual_exits)
        self.manual_exits = []
        print(f" {count} sortie(s) effacée(s)")
        return count
    
    def detect_exits(self, frame):
        """
        Détecte les sorties dans l'image
        Retourne toutes les sorties (manuelles + automatiques)
        """
        import time

        # Toujours inclure les sorties manuelles
        all_exits = self.manual_exits.copy()
        
        # Détection automatique si activée
        if self.auto_detection_enabled:
            current_time = time.time()
            
            if current_time - self.last_detection_time >= self.detection_interval:
                self.last_detection_time = current_time
                auto_exits = self._detect_green_signs(frame)
                all_exits.extend(auto_exits)
        
        self.exit_signs = all_exits
        return self.exit_signs
    
    def _detect_green_signs(self, frame):
        """
        Détection automatique des panneaux verts (EXIT)
        Utilise la détection de couleur HSV
        """
        detected = []
        
        # Conversion en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Plage de vert pour panneaux EXIT
        lower_green = np.array([45, 80, 80])
        upper_green = np.array([75, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Nettoyage du masque
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrage par taille
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Panneaux EXIT sont rectangulaires horizontaux
                if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
                    # Dans le haut de l'image (sorties souvent en hauteur)
                    if y < frame.shape[0] * 0.7:
                        detected.append({
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2),
                            'area': area,
                            'label': 'SORTIE (auto)',
                            'type': 'auto',
                            'confidence': min(area / self.max_area, 1.0),
                            'source': 'detection'
                        })
        
        return detected
    
    def draw_exits(self, frame, show_labels=True, show_distances=False, person_position=None):
        """
        Dessine les sorties sur l'image
        
        Args:
            frame: Image à annoter
            show_labels: Afficher les labels
            show_distances: Afficher les distances
            person_position: Position de la personne pour calcul distance
        """
        for i, exit_sign in enumerate(self.exit_signs):
            x, y, w, h = exit_sign['bbox']
            center = exit_sign['center']
            label = exit_sign.get('label', 'SORTIE')
            exit_type = exit_sign.get('type', 'unknown')
            
            # Couleur selon le type
            if exit_type == 'manual':
                color = (0, 255, 0)  # Vert
            elif exit_type == 'entrance':
                color = (255, 0, 0)  # Bleu
            elif exit_type == 'emergency':
                color = (0, 0, 255)  # Rouge
            elif exit_type == 'auto':
                color = (0, 200, 200)  # Jaune
            else:
                color = (255, 255, 255)  # Blanc
            
            # Rectangle autour de la sortie
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Point au centre
            cv2.circle(frame, center, 10, color, -1)
            cv2.circle(frame, center, 12, (255, 255, 255), 2)
            
            # Numéro
            cv2.putText(frame, str(i+1), center,
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, str(i+1), center,
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            # Label avec fond
            if show_labels:
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x, y - 40), (x + label_size[0] + 15, y - 5), color, -1)
                cv2.putText(frame, label, (x + 5, y - 15),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
            
            # Distance si personne détectée
            if show_distances and person_position:
                distance = np.linalg.norm(np.array(center) - np.array(person_position))
                distance_m = distance / 100  # Approximation en mètres
                
                # Ligne vers la personne
                cv2.line(frame, person_position, center, color, 2, cv2.LINE_AA)
                
                # Texte de distance
                mid_point = ((person_position[0] + center[0]) // 2,
                           (person_position[1] + center[1]) // 2)
                
                cv2.putText(frame, f"{distance_m:.1f}m", mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def get_nearest_exit(self, position):
        """
        Trouve la sortie la plus proche d'une position donnée
        
        Returns:
            dict: Informations sur la sortie la plus proche
        """
        if not self.exit_signs:
            return None
        
        min_distance = float('inf')
        nearest_exit = None
        
        for exit_sign in self.exit_signs:
            center = exit_sign['center']
            distance = np.sqrt(
                (center[0] - position[0])**2 + 
                (center[1] - position[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_exit = exit_sign.copy()
                nearest_exit['distance_px'] = distance
                nearest_exit['distance_m'] = distance / 100  # Approximation
        
        return nearest_exit
    
    def calculate_direction(self, from_pos, to_pos):
        """
        Calcule la direction et l'angle vers une sortie
        
        Returns:
            dict: {'direction': str, 'angle': float, 'distance': float}
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Angle en degrés
        angle = np.degrees(np.arctan2(-dy, dx))
        if angle < 0:
            angle += 360
        
        # Direction cardinale
        directions = [
            (0, "à droite"),
            (45, "devant à droite"),
            (90, "devant"),
            (135, "devant à gauche"),
            (180, "à gauche"),
            (225, "derrière à gauche"),
            (270, "derrière"),
            (315, "derrière à droite"),
            (360, "à droite")
        ]
        
        min_diff = 360
        direction_text = "devant"
        
        for deg, text in directions:
            diff = abs(angle - deg)
            if diff < min_diff:
                min_diff = diff
                direction_text = text
        
        # Distance
        distance = np.linalg.norm(np.array(to_pos) - np.array(from_pos))
        
        return {
            'direction': direction_text,
            'angle': angle,
            'distance_px': distance,
            'distance_m': distance / 100
        }
    
    def save_exits(self, filename="exits_config.json"):
        """Sauvegarde les sorties dans un fichier JSON"""
        data = {
            'manual_exits': self.manual_exits,
            'auto_detection_enabled': self.auto_detection_enabled,
            'saved_at': datetime.now().isoformat(),
            'count': len(self.manual_exits)
        }
        
        filepath = Path("output") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Sorties sauvegardées: {filepath}")
        return str(filepath)
    
    def load_exits(self, filename="exits_config.json"):
        """Charge les sorties depuis un fichier JSON"""
        filepath = Path("output") / filename
        
        if not filepath.exists():
            print(f"⚠️ Fichier non trouvé: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.manual_exits = data.get('manual_exits', [])
            self.auto_detection_enabled = data.get('auto_detection_enabled', False)
            
            print(f"📂 {len(self.manual_exits)} sortie(s) chargée(s)")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return False
    
    def get_exits_info(self):
        """Retourne des infos sur les sorties"""
        return {
            'total': len(self.exit_signs),
            'manual': len([e for e in self.exit_signs if e.get('source') == 'manual']),
            'auto': len([e for e in self.exit_signs if e.get('source') == 'detection']),
            'auto_detection_enabled': self.auto_detection_enabled
        }
    
    def draw_definition_overlay(self, frame):
        """Dessine un overlay pour le mode définition"""
        overlay = frame.copy()
        
        # Bandeau en haut
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Instructions
        cv2.putText(frame, "MODE DEFINITION SORTIE - Cliquez pour placer", 
                   (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "E=Sortie | I=Entree | U=Urgence | Echap=Annuler", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Curseur temporaire
        if self.temp_exit_position:
            x, y = self.temp_exit_position
            cv2.circle(frame, (x, y), 15, (0, 255, 0), 3)
            cv2.line(frame, (x-20, y), (x+20, y), (0, 255, 0), 2)
            cv2.line(frame, (x, y-20), (x, y+20), (0, 255, 0), 2)
        
        return frame


class ExitManager:
    """Gestionnaire de haut niveau pour les sorties"""
    
    def __init__(self):
        self.detector = ExitDetector()
        self.selected_exit = None
        self.guidance_history = []
    
    def process_frame(self, frame, person_position=None, show_guidance=True):
        """
        Traite une frame avec détection et affichage des sorties
        
        Args:
            frame: Image à traiter
            person_position: Position de la personne détectée
            show_guidance: Afficher le guidage
        
        Returns:
            frame annoté
        """
        # Détecter les sorties
        exits = self.detector.detect_exits(frame)
        
        # Dessiner les sorties
        frame = self.detector.draw_exits(
            frame, 
            show_labels=True,
            show_distances=(person_position is not None),
            person_position=person_position
        )
        
        # Guidage si personne détectée
        if person_position and exits and show_guidance:
            nearest = self.detector.get_nearest_exit(person_position)
            
            if nearest:
                direction_info = self.detector.calculate_direction(
                    person_position, 
                    nearest['center']
                )
                
                # Afficher les instructions
                self._draw_guidance(frame, direction_info, nearest)
        
        # Mode définition
        if self.detector.definition_mode:
            frame = self.detector.draw_definition_overlay(frame)
        
        # Statistiques
        self._draw_stats(frame)
        
        return frame
    
    def _draw_guidance(self, frame, direction_info, exit_info):
        """Dessine les instructions de guidage"""
        h, w = frame.shape[:2]
        
        # Bandeau de guidage
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texte de guidage
        direction = direction_info['direction']
        distance = direction_info['distance_m']
        
        guidance_text = f"🚪 Sortie {direction} - {distance:.1f}m"
        
        cv2.putText(frame, guidance_text, (20, h-50),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        
        # Flèche directionnelle
        angle = direction_info['angle']
        arrow_center = (w - 70, h - 50)
        arrow_length = 40
        
        end_x = int(arrow_center[0] + arrow_length * np.cos(np.radians(angle)))
        end_y = int(arrow_center[1] - arrow_length * np.sin(np.radians(angle)))
        
        cv2.arrowedLine(frame, arrow_center, (end_x, end_y),
                       (0, 255, 255), 3, tipLength=0.4)
    
    def _draw_stats(self, frame):
        """Dessine les statistiques des sorties"""
        info = self.detector.get_exits_info()
        
        stats_text = [
            f"Sorties: {info['total']}",
            f"Manuelles: {info['manual']}",
            f"Auto: {info['auto']}"
        ]
        
        y_offset = frame.shape[0] - 150
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25