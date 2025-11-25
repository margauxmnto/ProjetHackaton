# -*- coding: utf-8 -*-
"""
Système Unifié de Détection de Personnes pour Drone
Fusion de 3 approches: YOLO+DeepSORT, MediaPipe Pose, Motion Tracking
Avec interface graphique complète
"""

import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
# Import des modules de détection
from detection_modules import (FusionDetector, MediaPipePoseDetector,
                               MotionTracker, YOLODeepSORTDetector)
# Import du module de gestion des sorties
from exit_detection_module import ExitDetector, ExitManager
from PIL import Image, ImageTk


class DroneDetectionGUI:
    """Interface graphique principale pour le système de détection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Person Detection System - Fusion Multi-Modèles")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Variables de contrôle
        self.is_running = False
        self.is_paused = False
        self.is_recording = False
        self.current_frame = None
        self.video_capture = None
        self.video_writer = None
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Configuration
        self.config = {
            'source_type': tk.StringVar(value='webcam'),
            'source_path': tk.StringVar(value='0'),
            'detection_mode': tk.StringVar(value='fusion'),
            'output_dir': tk.StringVar(value='output'),
            'auto_record': tk.BooleanVar(value=False),
            'show_skeleton': tk.BooleanVar(value=True),
            'show_trajectory': tk.BooleanVar(value=True),
            'confidence_threshold': tk.DoubleVar(value=0.5),
        }
        
        # Statistiques
        self.stats = {
            'fps': 0,
            'persons_detected': 0,
            'total_detections': 0,
            'alerts': 0,
            'frame_count': 0
        }
        
        # Détecteurs
        self.detectors = {}
        self.current_detector = None
        
        # Gestionnaire de sorties
        self.exit_manager = ExitManager()
        self.exit_definition_mode = False
        self.current_exit_type = 'manual'  # Type de sortie à placer
        
        # Créer l'interface
        self.create_interface()
        
        # Initialiser les détecteurs
        self.initialize_detectors()
        
    def create_interface(self):
        """Crée l'interface utilisateur complète"""
        
        # ====== PANNEAU SUPÉRIEUR - CONTRÔLES ======
        top_frame = tk.Frame(self.root, bg='#1e1e1e', pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Section Source
        source_frame = tk.LabelFrame(top_frame, text="Source Vidéo", 
                                     bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        source_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(source_frame, text="Webcam", variable=self.config['source_type'],
                      value='webcam', bg='#1e1e1e', fg='white', 
                      selectcolor='#3e3e3e', command=self.on_source_change).pack(anchor=tk.W)
        tk.Radiobutton(source_frame, text="Téléphone (IVCam)", variable=self.config['source_type'],
                      value='phone', bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e', command=self.on_source_change).pack(anchor=tk.W)
        tk.Radiobutton(source_frame, text="Fichier Vidéo", variable=self.config['source_type'],
                      value='file', bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e', command=self.on_source_change).pack(anchor=tk.W)
        
        # Sélecteur d'index de caméra
        cam_index_frame = tk.Frame(source_frame, bg='#1e1e1e')
        cam_index_frame.pack(fill=tk.X, pady=5)
        tk.Label(cam_index_frame, text="Index/URL:", bg='#1e1e1e', fg='white').pack(side=tk.LEFT)
        self.camera_index = tk.Entry(cam_index_frame, width=20,
                                     bg='#3e3e3e', fg='white', insertbackground='white')
        self.camera_index.insert(0, "0")
        self.camera_index.pack(side=tk.LEFT, padx=5)
        
        # Aide RTSP
        tk.Label(source_frame, text="💡 RTSP: rtsp://IP:8080/video", 
                bg='#1e1e1e', fg='#888888', font=('Arial', 7)).pack()
        
        tk.Button(source_frame, text="🔍 Test Caméra", command=self.test_camera,
                 bg='#4a4a4a', fg='white', width=12).pack(pady=2)
        tk.Button(source_frame, text="📁 Parcourir", command=self.browse_video,
                 bg='#4a4a4a', fg='white', width=12).pack(pady=2)
        
        # Section Modèle
        model_frame = tk.LabelFrame(top_frame, text="Modèle de Détection",
                                   bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        model_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(model_frame, text="🔥 YOLO + DeepSORT", 
                      variable=self.config['detection_mode'],
                      value='yolo', bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e').pack(anchor=tk.W)
        tk.Radiobutton(model_frame, text="🧍 MediaPipe Pose",
                      variable=self.config['detection_mode'],
                      value='mediapipe', bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e').pack(anchor=tk.W)
        tk.Radiobutton(model_frame, text="🎯 Motion Tracking",
                      variable=self.config['detection_mode'],
                      value='motion', bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e').pack(anchor=tk.W)
        tk.Radiobutton(model_frame, text="⚡ FUSION (Tous combinés)",
                      variable=self.config['detection_mode'],
                      value='fusion', bg='#1e1e1e', fg='white',
                      selectcolor='#3e3e3e').pack(anchor=tk.W)
        
        # Section Options
        options_frame = tk.LabelFrame(top_frame, text="Options",
                                     bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        options_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Checkbutton(options_frame, text="Squelette", variable=self.config['show_skeleton'],
                      bg='#1e1e1e', fg='white', selectcolor='#3e3e3e').pack(anchor=tk.W)
        tk.Checkbutton(options_frame, text="Trajectoires", variable=self.config['show_trajectory'],
                      bg='#1e1e1e', fg='white', selectcolor='#3e3e3e').pack(anchor=tk.W)
        tk.Checkbutton(options_frame, text="Enreg. Auto", variable=self.config['auto_record'],
                      bg='#1e1e1e', fg='white', selectcolor='#3e3e3e').pack(anchor=tk.W)
        
        tk.Label(options_frame, text="Confiance:", bg='#1e1e1e', fg='white').pack(anchor=tk.W)
        tk.Scale(options_frame, from_=0.1, to=1.0, resolution=0.1,
                orient=tk.HORIZONTAL, variable=self.config['confidence_threshold'],
                bg='#3e3e3e', fg='white', troughcolor='#2e2e2e',
                highlightbackground='#1e1e1e').pack(fill=tk.X)
        
        # Section Contrôles
        control_frame = tk.LabelFrame(top_frame, text="Contrôles",
                                     bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        control_frame.pack(side=tk.LEFT, padx=10)
        
        self.btn_start = tk.Button(control_frame, text="▶️ Démarrer", 
                                   command=self.start_detection,
                                   bg='#00aa00', fg='white', width=12, height=2)
        self.btn_start.pack(pady=2)
        
        self.btn_pause = tk.Button(control_frame, text="⏸️ Pause",
                                   command=self.toggle_pause,
                                   bg='#ff9900', fg='white', width=12, state=tk.DISABLED)
        self.btn_pause.pack(pady=2)
        
        self.btn_stop = tk.Button(control_frame, text="⏹️ Stop",
                                  command=self.stop_detection,
                                  bg='#cc0000', fg='white', width=12, state=tk.DISABLED)
        self.btn_stop.pack(pady=2)
        
        # Section Enregistrement
        record_frame = tk.LabelFrame(top_frame, text="Enregistrement",
                                    bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        record_frame.pack(side=tk.LEFT, padx=10)
        
        self.btn_record = tk.Button(record_frame, text="🔴 Enregistrer",
                                    command=self.toggle_recording,
                                    bg='#cc0000', fg='white', width=12)
        self.btn_record.pack(pady=2)
        
        tk.Button(record_frame, text="📸 Capture", command=self.capture_frame,
                 bg='#4a4a4a', fg='white', width=12).pack(pady=2)
        
        tk.Button(record_frame, text="📊 Rapport", command=self.generate_report,
                 bg='#4a4a4a', fg='white', width=12).pack(pady=2)
        
        # Section Sorties de Secours
        exits_frame = tk.LabelFrame(top_frame, text="Sorties de Secours",
                                   bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        exits_frame.pack(side=tk.LEFT, padx=10)
        
        self.btn_exit_mode = tk.Button(exits_frame, text="🚪 Définir Sortie",
                                       command=self.toggle_exit_mode,
                                       bg='#4a4a4a', fg='white', width=12)
        self.btn_exit_mode.pack(pady=2)
        
        tk.Button(exits_frame, text="🔍 Auto Détection",
                 command=self.toggle_auto_detection,
                 bg='#4a4a4a', fg='white', width=12).pack(pady=2)
        
        tk.Button(exits_frame, text="🗑️ Suppr. Dernière",
                 command=self.remove_last_exit,
                 bg='#4a4a4a', fg='white', width=12).pack(pady=2)
        
        tk.Button(exits_frame, text="💾 Sauvegarder",
                 command=self.save_exits,
                 bg='#4a4a4a', fg='white', width=12).pack(pady=2)
        
        # ====== PANNEAU CENTRAL - AFFICHAGE VIDÉO ======
        center_frame = tk.Frame(self.root, bg='#2b2b2b')
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas pour la vidéo
        self.canvas = tk.Canvas(center_frame, bg='black', width=1280, height=720)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events pour définir les sorties
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)
        
        # ====== PANNEAU DROIT - STATISTIQUES ======
        stats_frame = tk.LabelFrame(center_frame, text="Statistiques en Temps Réel",
                                   bg='#1e1e1e', fg='white', font=('Arial', 11, 'bold'),
                                   width=250)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        stats_frame.pack_propagate(False)
        
        # Labels de statistiques
        self.stat_labels = {}
        stat_items = [
            ('FPS', 'fps_label'),
            ('Personnes Détectées', 'persons_label'),
            ('Détections Totales', 'total_label'),
            ('Alertes', 'alerts_label'),
            ('Frames', 'frames_label'),
            ('Mode', 'mode_label'),
            ('Statut', 'status_label')
        ]
        
        for i, (label, key) in enumerate(stat_items):
            frame = tk.Frame(stats_frame, bg='#1e1e1e')
            frame.pack(fill=tk.X, pady=5, padx=10)
            
            tk.Label(frame, text=f"{label}:", bg='#1e1e1e', fg='#aaaaaa',
                    font=('Arial', 9)).pack(anchor=tk.W)
            self.stat_labels[key] = tk.Label(frame, text="0", bg='#1e1e1e',
                                            fg='white', font=('Arial', 12, 'bold'))
            self.stat_labels[key].pack(anchor=tk.W)
        
        # Zone de log
        log_label = tk.Label(stats_frame, text="Journal d'Événements:",
                           bg='#1e1e1e', fg='white', font=('Arial', 10, 'bold'))
        log_label.pack(pady=(20, 5))
        
        self.log_text = tk.Text(stats_frame, bg='#0a0a0a', fg='#00ff00',
                               font=('Courier', 8), height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
        # ====== PANNEAU INFÉRIEUR - CONTRÔLES VIDÉO ======
        bottom_frame = tk.Frame(self.root, bg='#1e1e1e', pady=5)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.progress = ttk.Progressbar(bottom_frame, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        self.time_label = tk.Label(bottom_frame, text="00:00 / 00:00",
                                  bg='#1e1e1e', fg='white')
        self.time_label.pack()
        
    def initialize_detectors(self):
        """Initialise tous les détecteurs"""
        self.log("🔧 Initialisation des détecteurs...")
        
        try:
            # YOLO + DeepSORT
            self.log("   📦 Chargement YOLO + DeepSORT...")
            self.detectors['yolo'] = YOLODeepSORTDetector(
                confidence_threshold=self.config['confidence_threshold'].get()
            )
            
            # MediaPipe
            self.log("   📦 Chargement MediaPipe Pose...")
            self.detectors['mediapipe'] = MediaPipePoseDetector()
            
            # Motion Tracking
            self.log("   📦 Chargement Motion Tracker...")
            self.detectors['motion'] = MotionTracker()
            
            # Fusion
            self.log("   📦 Création du détecteur fusionné...")
            self.detectors['fusion'] = FusionDetector(
                self.detectors['yolo'],
                self.detectors['mediapipe'],
                self.detectors['motion']
            )
            
            self.log("✅ Tous les détecteurs sont prêts!")
            
        except Exception as e:
            self.log(f"❌ Erreur initialisation: {e}")
            messagebox.showerror("Erreur", f"Impossible d'initialiser les détecteurs:\n{e}")
    
    def on_source_change(self):
        """Appelé quand la source vidéo change"""
        source_type = self.config['source_type'].get()
        
        if source_type == 'webcam':
            index = self.camera_index.get()
            self.config['source_path'].set(str(index))
        elif source_type == 'phone':
            # IVCam utilise généralement l'index 1 ou une URL RTSP
            self.config['source_path'].set('1')
        elif source_type == 'file':
            self.config['source_path'].set('')
    
    def test_camera(self):
        """Teste si la caméra fonctionne"""
        source_input = self.camera_index.get().strip()
        
        # Déterminer si c'est un index ou une URL
        if source_input.startswith('rtsp://') or source_input.startswith('http://'):
            # C'est une URL
            source = source_input
            self.log(f"🔍 Test URL : {source}")
        else:
            # C'est un index
            try:
                source = int(source_input)
                self.log(f"🔍 Test caméra index {source}...")
            except ValueError:
                messagebox.showerror("Erreur", "Index invalide. Utilisez un nombre (0, 1, 2...) ou une URL RTSP")
                return
        
        test_cap = cv2.VideoCapture(source)
        
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                
                # Afficher un aperçu
                preview = frame.copy()
                if w > 640:
                    scale = 640 / w
                    preview = cv2.resize(preview, (int(w*scale), int(h*scale)))
                
                cv2.putText(preview, f"Source: {source}", (10, 30),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview, f"{w}x{h}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Aperçu - Appuyez sur une touche', preview)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                
                self.log(f"✅ Source {source} : FONCTIONNE !")
                messagebox.showinfo("Succès", 
                    f"Source détectée avec succès !\n\n"
                    f"Résolution : {w}x{h}\n"
                    f"Type : {'URL RTSP' if isinstance(source, str) and 'rtsp' in source else 'Webcam'}\n\n"
                    f"Vous pouvez maintenant cliquer sur 'Démarrer'")
                
                # Mettre à jour la config
                self.config['source_path'].set(str(source))
            else:
                self.log(f"⚠️ Source {source} : Ouverte mais pas d'image")
                messagebox.showwarning("Attention",
                    f"Source {source} est détectée mais ne retourne pas d'image.\n\n"
                    f"Solutions :\n"
                    f"• Pour IVCam : Vérifiez que l'app est bien connectée\n"
                    f"• Fermez les autres applications utilisant la caméra\n"
                    f"• Redémarrez IVCam Client sur PC\n"
                    f"• Essayez un autre index (0, 1, 2...)")
            test_cap.release()
        else:
            self.log(f"❌ Source {source} : Non disponible")
            
            if isinstance(source, int):
                error_msg = (
                    f"Impossible d'ouvrir la caméra {source}.\n\n"
                    f"Pour IVCam :\n"
                    f"1. Lancez 'IVCam Client' sur PC\n"
                    f"2. Lancez l'app IVCam sur téléphone\n"
                    f"3. Vérifiez qu'ils sont connectés (voyant vert)\n"
                    f"4. Essayez les index 0, 1, 2, 3...\n\n"
                    f"OU utilisez le mode RTSP :\n"
                    f"rtsp://IP_DU_TELEPHONE:8080/video"
                )
            else:
                error_msg = (
                    f"Impossible de se connecter à l'URL.\n\n"
                    f"Vérifiez :\n"
                    f"• L'adresse IP est correcte\n"
                    f"• Le port est correct (généralement 8080)\n"
                    f"• Le téléphone et PC sont sur le même réseau\n"
                    f"• Le pare-feu n'est pas bloqué"
                )
            
            messagebox.showerror("Erreur", error_msg)
    
    def browse_video(self):
        """Ouvre le dialogue pour sélectionner une vidéo"""
        filename = filedialog.askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[
                ("Vidéos", "*.mp4 *.avi *.mov *.mkv"),
                ("Tous fichiers", "*.*")
            ]
        )
        if filename:
            self.config['source_path'].set(filename)
            self.config['source_type'].set('file')
            self.log(f"📂 Vidéo sélectionnée: {os.path.basename(filename)}")
    
    def start_detection(self):
        """Démarre la détection"""
        if self.is_running:
            return
        
        # Obtenir la source depuis le champ Index/URL
        source_input = self.camera_index.get().strip()
        
        # Déterminer le type de source
        if source_input.startswith('rtsp://') or source_input.startswith('http://'):
            source = source_input  # C'est une URL
        elif source_input.isdigit():
            source = int(source_input)  # C'est un index
        else:
            messagebox.showerror("Erreur",
                "Source invalide.\n\n"
                "Utilisez :\n"
                "• Un index numérique (0, 1, 2...)\n"
                "• Une URL RTSP (rtsp://...)")
            return
        
        self.log(f"📹 Tentative d'ouverture : {source}")
        
        # Ouvrir la capture vidéo avec gestion d'erreur améliorée
        try:
            self.video_capture = cv2.VideoCapture(source)
            
            # Attendre un peu pour l'initialisation
            time.sleep(0.5)
            
            if not self.video_capture.isOpened():
                self.video_capture.release()
                error_msg = f"Impossible d'ouvrir la source vidéo : {source}\n\n"
                
                if isinstance(source, int):
                    error_msg += (
                        "Solutions :\n"
                        "1. Cliquez sur '🔍 Test Caméra' pour trouver le bon index\n"
                        "2. Essayez différents index (0, 1, 2...)\n"
                        "3. Vérifiez que la caméra n'est pas utilisée ailleurs\n"
                        "4. Redémarrez votre ordinateur\n"
                        "5. Vérifiez les permissions de la caméra"
                    )
                else:
                    error_msg += (
                        "Solutions :\n"
                        "1. Vérifiez que le fichier existe\n"
                        "2. Vérifiez le format (MP4, AVI, MOV...)\n"
                        "3. Essayez un autre fichier vidéo"
                    )
                
                messagebox.showerror("Erreur d'ouverture", error_msg)
                self.log(f"❌ Échec ouverture source : {source}")
                return
            
            # Tester la lecture d'une frame
            ret, test_frame = self.video_capture.read()
            if not ret or test_frame is None:
                self.video_capture.release()
                messagebox.showerror("Erreur",
                    f"Source ouverte mais impossible de lire les frames.\n\n"
                    f"La source pourrait être :\n"
                    f"• Utilisée par une autre application\n"
                    f"• Un fichier corrompu\n"
                    f"• Une caméra déconnectée")
                self.log(f"❌ Impossible de lire les frames")
                return
            
            # Remettre au début pour les fichiers vidéo
            if not isinstance(source, int):
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        except Exception as e:
            if self.video_capture:
                self.video_capture.release()
            messagebox.showerror("Erreur", f"Erreur lors de l'ouverture :\n{e}")
            self.log(f"❌ Exception : {e}")
            return
        
        # Sélectionner le détecteur
        mode = self.config['detection_mode'].get()
        self.current_detector = self.detectors[mode]
        
        # Enregistrement automatique
        if self.config['auto_record'].get():
            self.start_recording()
        
        # Mettre à jour l'interface
        self.is_running = True
        self.is_paused = False
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        
        self.log(f"▶️ Détection démarrée - Mode: {mode.upper()}")
        
        # Démarrer le thread de traitement
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
        
        # Démarrer l'affichage
        self.update_display()
    
    def process_video(self):
        """Thread de traitement vidéo"""
        frame_time_start = time.time()
        
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            ret, frame = self.video_capture.read()
            if not ret:
                self.log("⚠️ Fin de la vidéo ou erreur de lecture")
                self.root.after(0, self.stop_detection)
                break
            
            # Traiter la frame
            try:
                result = self.current_detector.detect(
                    frame,
                    show_skeleton=self.config['show_skeleton'].get(),
                    show_trajectory=self.config['show_trajectory'].get()
                )
                
                processed_frame = result['frame']
                detections = result['detections']
                
                # Ajouter la gestion des sorties
                person_position = None
                if detections:
                    # Prendre la première personne détectée
                    person_position = detections[0].get('center')
                
                # Traiter les sorties
                processed_frame = self.exit_manager.process_frame(
                    processed_frame,
                    person_position=person_position,
                    show_guidance=True
                )
                
                # Mettre à jour les statistiques
                self.stats['persons_detected'] = len(detections)
                self.stats['total_detections'] += len(detections)
                self.stats['frame_count'] += 1
                
                # Calculer FPS
                frame_time_end = time.time()
                self.stats['fps'] = 1.0 / (frame_time_end - frame_time_start)
                frame_time_start = frame_time_end
                
                # Enregistrer si actif
                if self.is_recording and self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # Ajouter à la queue d'affichage
                if not self.frame_queue.full():
                    self.frame_queue.put(processed_frame)
                
            except Exception as e:
                self.log(f"❌ Erreur traitement: {e}")
    
    def update_display(self):
        """Met à jour l'affichage de la vidéo"""
        if not self.is_running:
            return
        
        try:
            # Récupérer une frame de la queue
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Convertir pour Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Redimensionner si nécessaire
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    h, w = frame_rgb.shape[:2]
                    scale = min(canvas_width/w, canvas_height/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    # Sauvegarder le scale pour le calcul des coordonnées
                    self.display_scale = scale
                
                # Afficher
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.image = imgtk
                
                # Sauvegarder pour capture
                self.current_frame = frame
            
            # Mettre à jour les statistiques
            self.stat_labels['fps_label'].config(text=f"{self.stats['fps']:.1f}")
            self.stat_labels['persons_label'].config(text=str(self.stats['persons_detected']))
            self.stat_labels['total_label'].config(text=str(self.stats['total_detections']))
            self.stat_labels['alerts_label'].config(text=str(self.stats['alerts']))
            self.stat_labels['frames_label'].config(text=str(self.stats['frame_count']))
            self.stat_labels['mode_label'].config(text=self.config['detection_mode'].get().upper())
            self.stat_labels['status_label'].config(
                text="⏸️ PAUSE" if self.is_paused else "▶️ EN COURS"
            )
            
        except Exception as e:
            print(f"Erreur affichage: {e}")
        
        # Rappel
        self.root.after(30, self.update_display)
    
    def toggle_pause(self):
        """Met en pause / reprend la détection"""
        self.is_paused = not self.is_paused
        status = "⏸️ En pause" if self.is_paused else "▶️ Reprise"
        self.log(status)
        self.btn_pause.config(
            text="▶️ Reprendre" if self.is_paused else "⏸️ Pause"
        )
    
    def stop_detection(self):
        """Arrête la détection"""
        self.is_running = False
        self.is_paused = False
        
        if self.video_capture:
            self.video_capture.release()
        
        if self.is_recording:
            self.stop_recording()
        
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)
        
        self.log("⏹️ Détection arrêtée")
    
    def toggle_recording(self):
        """Démarre/arrête l'enregistrement"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Démarre l'enregistrement vidéo"""
        if not self.video_capture:
            messagebox.showwarning("Attention", "Démarrez d'abord la détection")
            return
        
        # Créer le dossier de sortie
        output_dir = Path(self.config['output_dir'].get())
        output_dir.mkdir(exist_ok=True)
        
        # Nom du fichier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = self.config['detection_mode'].get()
        filename = output_dir / f"recording_{mode}_{timestamp}.mp4"
        
        # Configurer le writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_writer = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))
        self.is_recording = True
        self.btn_record.config(text="⏹️ Arrêter Enreg.", bg='#009900')
        
        self.log(f"🔴 Enregistrement: {filename.name}")
    
    def stop_recording(self):
        """Arrête l'enregistrement vidéo"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        self.btn_record.config(text="🔴 Enregistrer", bg='#cc0000')
        self.log("⏹️ Enregistrement arrêté")
    
    def capture_frame(self):
        """Capture une image"""
        if not self.current_frame is None:
            output_dir = Path(self.config['output_dir'].get()) / "captures"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = output_dir / f"capture_{timestamp}.jpg"
            
            cv2.imwrite(str(filename), self.current_frame)
            self.log(f"📸 Capture: {filename.name}")
        else:
            messagebox.showwarning("Attention", "Aucune frame à capturer")
    
    def generate_report(self):
        """Génère un rapport JSON"""
        output_dir = Path(self.config['output_dir'].get()) / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'mode': self.config['detection_mode'].get(),
            'statistics': self.stats.copy(),
            'configuration': {k: v.get() for k, v in self.config.items()}
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log(f"📊 Rapport: {filename.name}")
        messagebox.showinfo("Succès", f"Rapport généré:\n{filename}")
    
    def log(self, message):
        """Ajoute un message au journal"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        print(log_message.strip())
    
    # ========== GESTION DES SORTIES ==========
    
    def toggle_exit_mode(self):
        """Active/désactive le mode définition de sortie"""
        self.exit_definition_mode = not self.exit_definition_mode
        self.exit_manager.detector.definition_mode = self.exit_definition_mode
        
        if self.exit_definition_mode:
            self.btn_exit_mode.config(text="✅ Mode Actif", bg='#00aa00')
            self.log("🚪 Mode définition ACTIVÉ - Cliquez pour placer des sorties")
            
            # Afficher dialogue de choix de type
            self.show_exit_type_dialog()
        else:
            self.btn_exit_mode.config(text="🚪 Définir Sortie", bg='#4a4a4a')
            self.log("🚪 Mode définition DÉSACTIVÉ")
    
    def show_exit_type_dialog(self):
        """Affiche un dialogue pour choisir le type de sortie"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Type de Sortie")
        dialog.geometry("300x200")
        dialog.configure(bg='#2b2b2b')
        
        tk.Label(dialog, text="Choisissez le type de sortie :",
                bg='#2b2b2b', fg='white', font=('Arial', 11)).pack(pady=10)
        
        exit_types = [
            ('🚪 Sortie Standard', 'manual', 'SORTIE'),
            ('🔵 Entrée', 'entrance', 'ENTRÉE'),
            ('🔴 Sortie d\'Urgence', 'emergency', 'URGENCE'),
            ('🟢 Sortie de Secours', 'evacuation', 'SECOURS')
        ]
        
        for label, type_key, type_label in exit_types:
            tk.Button(dialog, text=label,
                     command=lambda t=type_key, l=type_label: self.set_exit_type(dialog, t, l),
                     bg='#4a4a4a', fg='white', width=20, pady=5).pack(pady=5)
    
    def set_exit_type(self, dialog, exit_type, label):
        """Définit le type de sortie à placer"""
        self.current_exit_type = exit_type
        self.current_exit_label = label
        dialog.destroy()
        self.log(f"Type sélectionné : {label}")
    
    def on_canvas_click(self, event):
        """Gère le clic sur le canvas"""
        if not self.exit_definition_mode:
            return
        
        # Convertir les coordonnées canvas en coordonnées image
        canvas_x, canvas_y = event.x, event.y
        
        # Ratio de l'image affichée
        if not hasattr(self, 'display_scale'):
            return
        
        # Coordonnées réelles dans l'image
        img_x = int(canvas_x / self.display_scale)
        img_y = int(canvas_y / self.display_scale)
        
        # Ajouter la sortie
        self.exit_manager.detector.add_manual_exit(
            (img_x, img_y),
            label=self.current_exit_label,
            exit_type=self.current_exit_type
        )
        
        self.log(f"✅ {self.current_exit_label} placée à ({img_x}, {img_y})")
    
    def on_canvas_right_click(self, event):
        """Clic droit pour supprimer une sortie"""
        if not self.exit_definition_mode:
            return
        
        canvas_x, canvas_y = event.x, event.y
        
        if not hasattr(self, 'display_scale'):
            return
        
        img_x = int(canvas_x / self.display_scale)
        img_y = int(canvas_y / self.display_scale)
        
        # Supprimer la sortie proche
        removed = self.exit_manager.detector.remove_exit((img_x, img_y), tolerance=50)
        
        if removed:
            self.log(f"🗑️ Sortie supprimée")
        else:
            self.log(f"⚠️ Aucune sortie proche trouvée")
    
    def toggle_auto_detection(self):
        """Active/désactive la détection automatique"""
        enabled = self.exit_manager.detector.toggle_auto_detection()
        status = "ACTIVÉE" if enabled else "DÉSACTIVÉE"
        self.log(f"🔍 Détection automatique : {status}")
    
    def remove_last_exit(self):
        """Supprime la dernière sortie"""
        removed = self.exit_manager.detector.remove_last_exit()
        if removed:
            self.log(f"🗑️ Sortie supprimée : {removed['label']}")
        else:
            messagebox.showinfo("Info", "Aucune sortie à supprimer")
    
    def save_exits(self):
        """Sauvegarde les sorties"""
        filename = self.exit_manager.detector.save_exits()
        self.log(f"💾 Sorties sauvegardées : {filename}")
        messagebox.showinfo("Succès", f"Sorties sauvegardées :\n{filename}")
    
    def load_exits(self):
        """Charge les sorties"""
        success = self.exit_manager.detector.load_exits()
        if success:
            self.log("📂 Sorties chargées")
            messagebox.showinfo("Succès", "Sorties chargées avec succès")
        else:
            messagebox.showwarning("Attention", "Aucune sauvegarde trouvée")


def main():
    """Point d'entrée principal"""
    root = tk.Tk()
    app = DroneDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()