#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蚊子雷达追踪打击系统
Mosquito Radar Tracking System

Author: Killerzeno
Description: 使用计算机视觉和雷达可视化技术检测和追踪蚊子
"""

import os
import time
import math
import wave
import struct
import threading
import warnings
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod

import cv2
import numpy as np
import matplotlib
import pygame
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

# 过滤掉一些不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 设置后端
matplotlib.use('Qt5Agg')

# 设置中文字体
rcParams['font.family'] = 'Microsoft YaHei Mono'
rcParams['axes.unicode_minus'] = False


# ==================== 配置类 ====================
@dataclass
class AudioConfig:
    """音频配置"""
    sample_rate: int = 44100
    duration: float = 3.0
    frequency: int = 22000
    play_interval: int = 5


@dataclass
class CameraConfig:
    """摄像头配置"""
    width: int = 640
    height: int = 480
    buffer_size: int = 1


@dataclass
class DetectionConfig:
    """检测配置"""
    edge_threshold_low: int = 50
    edge_threshold_high: int = 150
    min_area: int = 5
    max_area: int = 150
    min_perimeter: int = 10
    min_radius: int = 2
    max_radius: int = 20
    min_circularity: float = 0.5
    max_tracking_distance: int = 50
    track_timeout: float = 0.5


@dataclass
class RadarConfig:
    """雷达配置"""
    max_distance: int = 500
    angles: int = 360
    scan_speed: int = 5
    animation_interval: int = 30


@dataclass
class SystemConfig:
    """系统配置"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)


# ==================== 数据类 ====================
@dataclass
class Position:
    """位置信息"""
    x: float
    y: float
    timestamp: float


@dataclass
class Detection:
    """检测结果"""
    position: Position
    angle: float
    distance: float
    area: float
    radius: float


@dataclass
class SystemStats:
    """系统统计信息"""
    total_detected: int = 0
    detected_today: int = 0
    true_positives: int = 0
    false_positives: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def accuracy(self) -> float:
        """计算检测精度"""
        total = self.true_positives + self.false_positives
        return (self.true_positives / total * 100) if total > 0 else 0.0


# ==================== 轨迹追踪类 ====================
class Track:
    """单个目标轨迹"""
    
    def __init__(self, track_id: int, initial_position: Position):
        self.id = track_id
        self.positions: List[Position] = [initial_position]
        self.speeds: List[float] = []
        self.directions: List[float] = []
        self.last_update_time = initial_position.timestamp
        self.is_active = True

    def update_position(self, new_position: Position) -> None:
        """更新位置信息"""
        self._calculate_motion_metrics(new_position)
        self.positions.append(new_position)
        self.last_update_time = new_position.timestamp
        self.is_active = True

    def _calculate_motion_metrics(self, new_position: Position) -> None:
        """计算运动指标"""
        last_pos = self.positions[-1]
        delta_x = new_position.x - last_pos.x
        delta_y = new_position.y - last_pos.y
        delta_time = new_position.timestamp - last_pos.timestamp
        
        if delta_time > 0:
            speed = np.sqrt(delta_x ** 2 + delta_y ** 2) / delta_time
            direction = np.degrees(np.arctan2(delta_y, delta_x))
            self.speeds.append(speed)
            self.directions.append(direction)

    @property
    def current_speed(self) -> float:
        """获取当前速度"""
        return np.mean(self.speeds[-3:]) if self.speeds else 0.0

    @property
    def current_direction(self) -> Optional[float]:
        """获取当前方向"""
        return self.directions[-1] if self.directions else None

    def is_timeout(self, current_time: float, timeout: float) -> bool:
        """检查是否超时"""
        return current_time - self.last_update_time > timeout


# ==================== 音频系统 ====================
class AudioSystem:
    """音频系统管理"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.mosquito_sound: Optional[pygame.mixer.Sound] = None
        self.is_enabled = True
        self.is_playing = False
        self.last_play_time = 0.0
        self._initialize()

    def _initialize(self) -> None:
        """初始化音频系统"""
        try:
            pygame.mixer.init()
            sound_file = self._generate_anti_mosquito_sound()
            self.mosquito_sound = pygame.mixer.Sound(sound_file)
            print("已生成驱蚊音频文件")
        except Exception as error:
            print(f"音频初始化失败: {error}")

    def _generate_anti_mosquito_sound(self) -> str:
        """生成驱蚊音频文件"""
        samples = []
        total_samples = int(self.config.duration * self.config.sample_rate)
        
        for i in range(total_samples):
            sample = 0.5 * math.sin(2 * math.pi * self.config.frequency * i / self.config.sample_rate)
            samples.append(sample)

        filename = "anti_mosquito_repellent.wav"
        with wave.open(filename, 'w') as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(self.config.sample_rate)
            for sample in samples:
                data = struct.pack('<h', int(sample * 32767))
                wave_file.writeframesraw(data)
        
        return filename

    def play_if_needed(self) -> None:
        """在需要时播放声音"""
        if not self.is_enabled or not self.mosquito_sound:
            return
            
        current_time = time.time()
        if current_time - self.last_play_time > self.config.play_interval:
            try:
                self.mosquito_sound.play()
                self.last_play_time = current_time
                self.is_playing = True
            except Exception as error:
                print(f"播放音频失败: {error}")

    def toggle_enabled(self) -> None:
        """切换音频开关"""
        self.is_enabled = not self.is_enabled

    def update_playing_status(self) -> None:
        """更新播放状态"""
        if self.is_playing and not pygame.mixer.get_busy():
            self.is_playing = False


# ==================== 摄像头管理 ====================
class CameraManager:
    """摄像头管理器"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture: Optional[cv2.VideoCapture] = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=16, detectShadows=False
        )
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """初始化摄像头"""
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("警告: 无法打开摄像头，尝试使用其他摄像头...")
                self.capture = cv2.VideoCapture(1)
            
            self._configure_camera()
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            self.capture = None

    def _configure_camera(self) -> None:
        """配置摄像头参数"""
        if self.capture:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

    def get_frame(self) -> np.ndarray:
        """获取摄像头帧"""
        if not self.capture or not self.capture.isOpened():
            return self._create_dummy_frame()
        
        success, frame = self.capture.read()
        if not success:
            return self._create_dummy_frame()
        
        return cv2.flip(frame, 1)

    def _create_dummy_frame(self) -> np.ndarray:
        """创建模拟帧"""
        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

    def release(self) -> None:
        """释放摄像头资源"""
        if self.capture and self.capture.isOpened():
            self.capture.release()


# ==================== 检测器 ====================
class MosquitoDetector:
    """蚊子检测器"""
    
    def __init__(self, config: DetectionConfig, camera_manager: CameraManager):
        self.config = config
        self.camera_manager = camera_manager

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """检测蚊子"""
        gray_frame = self._preprocess_frame(frame)
        foreground_mask = self._extract_foreground(gray_frame)
        contours = self._find_contours(foreground_mask)
        return self._analyze_contours(contours, frame)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _extract_foreground(self, gray_frame: np.ndarray) -> np.ndarray:
        """提取前景"""
        foreground_mask = self.camera_manager.background_subtractor.apply(gray_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        return foreground_mask

    def _find_contours(self, mask: np.ndarray) -> List:
        """查找轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _analyze_contours(self, contours: List, frame: np.ndarray) -> List[Detection]:
        """分析轮廓并生成检测结果"""
        detections = []
        current_time = time.time()

        for contour in contours:
            detection = self._analyze_single_contour(contour, frame, current_time)
            if detection:
                detections.append(detection)

        return detections

    def _analyze_single_contour(self, contour, frame: np.ndarray, timestamp: float) -> Optional[Detection]:
        """分析单个轮廓"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if not self._is_valid_size(area, perimeter):
            return None

        (x_pos, y_pos), radius = cv2.minEnclosingCircle(contour)
        
        if not self._is_valid_radius(radius):
            return None

        if not self._is_valid_circularity(area, perimeter):
            return None

        return self._create_detection(x_pos, y_pos, radius, area, frame, timestamp)

    def _is_valid_size(self, area: float, perimeter: float) -> bool:
        """检查尺寸是否有效"""
        return (self.config.min_area < area < self.config.max_area and 
                perimeter > self.config.min_perimeter)

    def _is_valid_radius(self, radius: float) -> bool:
        """检查半径是否有效"""
        return self.config.min_radius < radius < self.config.max_radius

    def _is_valid_circularity(self, area: float, perimeter: float) -> bool:
        """检查圆度是否有效"""
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        return circularity > self.config.min_circularity

    def _create_detection(self, x_pos: float, y_pos: float, radius: float, 
                         area: float, frame: np.ndarray, timestamp: float) -> Detection:
        """创建检测结果"""
        center_x = x_pos - frame.shape[1] // 2
        center_y = y_pos - frame.shape[0] // 2
        
        angle = np.arctan2(center_y, center_x) % (2 * np.pi)
        distance = min(np.sqrt(center_x ** 2 + center_y ** 2), self.config.max_tracking_distance)
        
        position = Position(x_pos, y_pos, timestamp)
        return Detection(position, angle, distance, area, radius)


# ==================== 追踪器 ====================
class MultiTargetTracker:
    """多目标追踪器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.stats = SystemStats()

    def update(self, detections: List[Detection]) -> List[Tuple[float, float, Track]]:
        """更新追踪状态"""
        current_time = time.time()
        self._mark_inactive_tracks(current_time)
        
        active_tracks = [track for track in self.tracks if track.is_active]
        matched_detections = self._match_detections_to_tracks(detections, active_tracks)
        self._create_new_tracks(detections, matched_detections, current_time)
        
        return self._get_track_info(detections, matched_detections)

    def _mark_inactive_tracks(self, current_time: float) -> None:
        """标记不活跃的轨迹"""
        for track in self.tracks:
            if track.is_timeout(current_time, self.config.track_timeout):
                track.is_active = False

    def _match_detections_to_tracks(self, detections: List[Detection], 
                                  active_tracks: List[Track]) -> List[bool]:
        """将检测结果匹配到现有轨迹"""
        matched = [False] * len(detections)

        for track in active_tracks:
            best_match_idx = self._find_best_match(track, detections, matched)
            if best_match_idx is not None:
                detection = detections[best_match_idx]
                track.update_position(detection.position)
                matched[best_match_idx] = True
                self.stats.true_positives += 1
            else:
                self.stats.false_positives += 1

        return matched

    def _find_best_match(self, track: Track, detections: List[Detection], 
                        matched: List[bool]) -> Optional[int]:
        """为轨迹找到最佳匹配的检测"""
        min_distance = float('inf')
        best_match_idx = None

        for i, detection in enumerate(detections):
            if matched[i]:
                continue
                
            last_pos = track.positions[-1]
            distance = self._calculate_distance(last_pos, detection.position)

            if distance < self.config.max_tracking_distance and distance < min_distance:
                min_distance = distance
                best_match_idx = i

        return best_match_idx

    def _calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """计算两个位置之间的距离"""
        return np.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)

    def _create_new_tracks(self, detections: List[Detection], 
                          matched: List[bool], current_time: float) -> None:
        """为未匹配的检测创建新轨迹"""
        for i, detection in enumerate(detections):
            if not matched[i]:
                new_track = Track(self.next_track_id, detection.position)
                self.tracks.append(new_track)
                self.next_track_id += 1
                self.stats.total_detected += 1
                self.stats.detected_today += 1

    def _get_track_info(self, detections: List[Detection], 
                       matched: List[bool]) -> List[Tuple[float, float, Track]]:
        """获取轨迹信息"""
        track_info = []
        detection_idx = 0
        
        for track in self.tracks:
            if track.is_active:
                # 找到对应的检测结果
                while detection_idx < len(detections) and matched[detection_idx]:
                    detection_idx += 1
                    
                if detection_idx < len(detections):
                    detection = detections[detection_idx]
                    track_info.append((detection.angle, detection.distance, track))
                    detection_idx += 1

        return track_info

    @property
    def active_track_count(self) -> int:
        """获取活跃轨迹数量"""
        return len([track for track in self.tracks if track.is_active])


# ==================== 雷达可视化 ====================
class RadarVisualizer:
    """雷达可视化器"""
    
    def __init__(self, config: RadarConfig):
        self.config = config
        self.current_scan_angle = 0
        self.figure = None
        self.radar_axis = None
        self.status_axis = None
        self.stats_axis = None
        self.scan_line = None
        self.mosquito_dots = None
        self.trail_lines = []
        self.status_texts = []
        self.stats_texts = []
        
        self._initialize_visualization()

    def _initialize_visualization(self) -> None:
        """初始化可视化界面"""
        plt.style.use('dark_background')
        self.figure = plt.figure(figsize=(9, 6), facecolor='black')
        self.figure.suptitle('蚊子雷达追踪系统', color='white', 
                           fontsize=14, fontweight='bold', y=0.95)

        self._create_layout()
        self._setup_radar_display()
        self._initialize_info_panels()

    def _create_layout(self) -> None:
        """创建布局"""
        gs = self.figure.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1], 
                                     hspace=0.25, wspace=0.15, 
                                     left=0.08, right=0.95, top=0.90, bottom=0.08)
        
        self.radar_axis = self.figure.add_subplot(gs[:, 0], polar=True, facecolor=(0, 0.05, 0))
        self.status_axis = self.figure.add_subplot(gs[0, 1], facecolor='black')
        self.stats_axis = self.figure.add_subplot(gs[1, 1], facecolor='black')
        
        self.status_axis.axis('off')
        self.stats_axis.axis('off')

    def _setup_radar_display(self) -> None:
        """设置雷达显示"""
        self.radar_axis.set_theta_zero_location('N')
        self.radar_axis.set_theta_direction(-1)
        self.radar_axis.set_ylim(0, self.config.max_distance)
        self.radar_axis.set_yticklabels([])
        self.radar_axis.grid(color='lime', alpha=0.3, linestyle='-', linewidth=0.5)
        self.radar_axis.tick_params(axis='both', colors='lime')
        
        self._setup_range_rings()
        self._initialize_radar_elements()

    def _setup_range_rings(self) -> None:
        """设置距离环"""
        ranges = [100, 200, 300, 400, 500]
        self.radar_axis.set_rticks(ranges)
        for radius in ranges:
            self.radar_axis.text(0, radius + 20, f'{radius}cm', color='lime', 
                               ha='center', va='center', fontsize=8, alpha=0.8)

    def _initialize_radar_elements(self) -> None:
        """初始化雷达元素"""
        self.center_point = self.radar_axis.scatter([0], [0], c='lime', s=20, alpha=0.8)
        self.scan_line = self.radar_axis.plot([], [], color='cyan', linestyle='-', 
                                            linewidth=2, alpha=0.8)[0]
        self.mosquito_dots = self.radar_axis.scatter([], [], c='red', s=25, alpha=0.9,
                                                   edgecolors='white', linewidths=1, zorder=10)

    def _initialize_info_panels(self) -> None:
        """初始化信息面板"""
        self._create_status_panel()
        self._create_stats_panel()

    def _create_status_panel(self) -> None:
        """创建状态面板"""
        self.status_axis.text(0.5, 0.9, '系统状态', color='cyan', fontsize=12, 
                             fontweight='bold', ha='center', transform=self.status_axis.transAxes)
        
        status_labels = ['运行状态:', '摄像头:', '声波攻击:', '扫描角度:']
        self.status_texts = []
        
        for i, label in enumerate(status_labels):
            self.status_axis.text(0.1, 0.7 - i*0.15, label, color='white', fontsize=9,
                                transform=self.status_axis.transAxes)
            text = self.status_axis.text(0.6, 0.7 - i*0.15, '', color='lime', fontsize=9,
                                       transform=self.status_axis.transAxes)
            self.status_texts.append(text)

    def _create_stats_panel(self) -> None:
        """创建统计面板"""
        self.stats_axis.text(0.5, 0.9, '检测统计', color='cyan', fontsize=12, 
                            fontweight='bold', ha='center', transform=self.stats_axis.transAxes)
        
        stats_labels = ['当前目标:', '今日检测:', '总计检测:', '检测精度:']
        self.stats_texts = []
        
        for i, label in enumerate(stats_labels):
            self.stats_axis.text(0.1, 0.7 - i*0.15, label, color='white', fontsize=9,
                               transform=self.stats_axis.transAxes)
            text = self.stats_axis.text(0.6, 0.7 - i*0.15, '', color='lime', fontsize=9,
                                      transform=self.stats_axis.transAxes)
            self.stats_texts.append(text)
        
        self.stats_axis.text(0.5, 0.1, '按键控制: A-声波开关 P-截图 Q-退出', 
                            color='gray', fontsize=8, ha='center',
                            transform=self.stats_axis.transAxes)

    def update_display(self, track_info: List[Tuple[float, float, Track]], 
                      frame: np.ndarray, audio_system: AudioSystem, 
                      stats: SystemStats) -> List:
        """更新显示"""
        self._update_scan_line()
        self._update_mosquito_markers(track_info)
        self._update_trail_lines(track_info, frame)
        self._update_info_displays(track_info, audio_system, stats)
        
        return self._get_artists()

    def _update_scan_line(self) -> None:
        """更新扫描线"""
        self.current_scan_angle = (self.current_scan_angle + self.config.scan_speed * 2) % 360
        scan_radians = np.radians(self.current_scan_angle)
        self.scan_line.set_data([scan_radians, scan_radians], [0, self.config.max_distance])

    def _update_mosquito_markers(self, track_info: List[Tuple[float, float, Track]]) -> None:
        """更新蚊子标记"""
        if track_info:
            angles, distances, tracks = zip(*track_info)
            sizes = [22] * len(angles)
            colors = ['red' if track.is_active else 'orange' for track in tracks]
            self.mosquito_dots.set_offsets(np.column_stack([angles, distances]))
            self.mosquito_dots.set_sizes(sizes)
            self.mosquito_dots.set_color(colors)
        else:
            self.mosquito_dots.set_offsets(np.empty((0, 2)))

    def _update_trail_lines(self, track_info: List[Tuple[float, float, Track]], 
                           frame: np.ndarray) -> None:
        """更新轨迹线"""
        self._clear_trail_lines()
        
        for angle, distance, track in track_info:
            if len(track.positions) > 1:
                trail_data = self._calculate_trail_data(track, frame)
                if trail_data:
                    line = self.radar_axis.plot(trail_data['angles'], trail_data['distances'],
                                               color='yellow', alpha=0.7,
                                               linewidth=1, zorder=5)[0]
                    self.trail_lines.append(line)

    def _clear_trail_lines(self) -> None:
        """清除轨迹线"""
        for line in self.trail_lines:
            line.remove()
        self.trail_lines = []

    def _calculate_trail_data(self, track: Track, frame: np.ndarray) -> Optional[Dict]:
        """计算轨迹数据"""
        history_angles = []
        history_distances = []
        
        # 只显示最近3个位置的轨迹
        recent_positions = track.positions[-3:]
        
        for position in recent_positions:
            center_x = position.x - frame.shape[1] // 2
            center_y = position.y - frame.shape[0] // 2
            angle = np.arctan2(center_y, center_x) % (2 * np.pi)
            distance = np.sqrt(center_x ** 2 + center_y ** 2)
            history_angles.append(angle)
            history_distances.append(distance)

        if len(history_angles) > 1:
            return {'angles': history_angles, 'distances': history_distances}
        
        return None

    def _update_info_displays(self, track_info: List[Tuple[float, float, Track]], 
                             audio_system: AudioSystem, stats: SystemStats) -> None:
        """更新信息显示"""
        self._update_status_display(audio_system)
        self._update_stats_display(track_info, stats)

    def _update_status_display(self, audio_system: AudioSystem) -> None:
        """更新状态显示"""
        self.status_texts[0].set_text("运行中")
        self.status_texts[1].set_text("开启")
        
        sound_status = "开启" if audio_system.is_enabled else "关闭"
        if audio_system.is_playing:
            sound_status += " (播放中)"
        self.status_texts[2].set_text(sound_status)
        self.status_texts[3].set_text(f"{self.current_scan_angle:.0f}°")

    def _update_stats_display(self, track_info: List[Tuple[float, float, Track]], 
                             stats: SystemStats) -> None:
        """更新统计显示"""
        active_count = len(track_info)
        self.stats_texts[0].set_text(f"{active_count}")
        self.stats_texts[1].set_text(f"{stats.detected_today}")
        self.stats_texts[2].set_text(f"{stats.total_detected}")
        self.stats_texts[3].set_text(f"{stats.accuracy:.1f}%")

    def _get_artists(self) -> List:
        """获取需要更新的艺术家对象"""
        artists = [self.scan_line, self.mosquito_dots]
        artists.extend(self.status_texts)
        artists.extend(self.stats_texts)
        artists.extend(self.trail_lines)
        return artists

    def take_screenshot(self, screenshot_count: int) -> None:
        """截取屏幕截图"""
        screenshot_dir = "screenshots"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshot_dir}/screenshot_{screenshot_count}_{timestamp}.png"
        plt.savefig(filename)
        print(f"截图已保存: {filename}")


# ==================== 主系统类 ====================
class MosquitoRadarTracker:
    """蚊子雷达追踪器主类"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.screenshot_count = 0
        
        # 初始化各个子系统
        self.audio_system = AudioSystem(self.config.audio)
        self.camera_manager = CameraManager(self.config.camera)
        self.detector = MosquitoDetector(self.config.detection, self.camera_manager)
        self.tracker = MultiTargetTracker(self.config.detection)
        self.visualizer = RadarVisualizer(self.config.radar)
        
        self.animation = None

    def update_frame(self, frame_number: int) -> List:
        """更新帧数据（动画回调函数）"""
        # 获取摄像头帧
        frame = self.camera_manager.get_frame()
        
        # 检测蚊子
        detections = self.detector.detect(frame)
        
        # 更新追踪
        track_info = self.tracker.update(detections)
        
        # 播放驱蚊声音
        if track_info:
            self.audio_system.play_if_needed()
        
        # 更新可视化
        artists = self.visualizer.update_display(
            track_info, frame, self.audio_system, self.tracker.stats
        )
        
        # 处理输入和显示
        self._handle_frame_display(frame)
        if not self._handle_user_input():
            return []
        
        # 更新音频状态
        self.audio_system.update_playing_status()
        
        return artists

    def _handle_frame_display(self, frame: np.ndarray) -> None:
        """处理帧显示"""
        # 边缘检测显示
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred, self.config.detection.edge_threshold_low, 
                         self.config.detection.edge_threshold_high)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_display = np.zeros_like(frame)
        cv2.drawContours(edge_display, contours, -1, (0, 255, 0), 1)

        # 显示窗口
        cv2.imshow('Mosquito Tracking', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow('Edges', edge_display)

    def _handle_user_input(self) -> bool:
        """处理用户输入"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self._stop_animation()
            return False
        elif key == ord('a'):
            self.audio_system.toggle_enabled()
        elif key == ord('p'):
            self._take_screenshot()
        
        return True

    def _stop_animation(self) -> None:
        """停止动画"""
        if self.animation:
            self.animation.event_source.stop()
        self.cleanup()

    def _take_screenshot(self) -> None:
        """截取屏幕截图"""
        self.visualizer.take_screenshot(self.screenshot_count)
        self.screenshot_count += 1

    def cleanup(self) -> None:
        """清理资源"""
        self.camera_manager.release()
        cv2.destroyAllWindows()
        plt.close('all')

    def run(self) -> None:
        """运行主程序"""
        try:
            # 启动时播放一次声波
            if self.audio_system.is_enabled and self.audio_system.mosquito_sound:
                self.audio_system.mosquito_sound.play()
                self.audio_system.is_playing = True
                self.audio_system.last_play_time = time.time()

            # 创建并启动动画
            self.animation = FuncAnimation(
                self.visualizer.figure, 
                self.update_frame, 
                frames=None, 
                interval=self.config.radar.animation_interval, 
                blit=True, 
                cache_frame_data=False
            )
            
            plt.show()
            
        except Exception as error:
            print(f"程序错误: {error}")
        finally:
            self.cleanup()


# ==================== 主程序入口 ====================
def main():
    """主程序入口"""
    print("正在启动蚊子雷达追踪打击系统...")
    tracker = MosquitoRadarTracker()
    tracker.run()


if __name__ == "__main__":
    main()
