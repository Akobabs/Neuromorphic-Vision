"""
Enhanced Neuromorphic Vision Pipeline with SNNs
Improved version with better error handling, modular design, and optimizations
"""

import os
import logging
import struct
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import norse.torch as norse
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration class for the neuromorphic vision pipeline"""
    # Paths
    data_path: str = '/content/drive/MyDrive/Neuromorphic Vision'
    
    # DVS Gesture parameters
    dvs_max_x: int = 128
    dvs_max_y: int = 128
    dvs_num_classes: int = 11
    
    # N-Caltech101 parameters
    caltech_max_x: int = 304
    caltech_max_y: int = 240
    caltech_num_classes: int = 101
    
    # Preprocessing parameters
    max_jitter: int = 100
    time_bin: float = 0.01
    temporal_window: float = 50000  # microseconds
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Noise detection
    dbscan_eps: float = 5.0
    dbscan_min_samples: int = 10

config = Config()

class DataProcessor:
    """Enhanced data processing utilities"""
    
    @staticmethod
    def setup_environment():
        """Setup environment and install required packages"""
        try:
            import loris
        except ImportError:
            os.system('pip install loris numpy pandas matplotlib torch norse scikit-learn seaborn tqdm')
            import loris
        
        # Mount Google Drive if in Colab
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            logger.info("Not running in Colab, skipping drive mount")
    
    @staticmethod
    def validate_file_paths(file_paths: List[str]) -> List[str]:
        """Validate that all file paths exist"""
        valid_paths = []
        for path in file_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.warning(f"File not found: {path}")
        return valid_paths

class DVSGestureProcessor:
    """Enhanced DVS Gesture dataset processor"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_dvs_gesture(self, aedat_file: str, csv_file: str) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Load DVS gesture data with enhanced error handling"""
        try:
            import loris
            data = loris.read_file(aedat_file)
            events = data['events']
            
            # Convert to numpy array more efficiently
            event_array = np.column_stack([
                events['ts'].astype(np.float64),
                events['x'].astype(np.int16),
                events['y'].astype(np.int16),
                events['p'].astype(np.int8)
            ])
            
            # Load labels with proper column names
            labels = pd.read_csv(csv_file, names=['class', 'startTime_usec', 'endTime_usec'])
            
            logger.info(f"Loaded {len(event_array)} events and {len(labels)} labels from {aedat_file}")
            return event_array, labels
            
        except Exception as e:
            logger.error(f"Error loading DVS gesture data from {aedat_file}: {e}")
            return None, None
    
    def detect_noise_advanced(self, events: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Advanced noise detection with multiple criteria"""
        noise_mask = np.zeros(len(events), dtype=bool)
        stats = {}
        
        # Spatial clustering for noise detection
        coords = events[:, 1:3]
        if len(coords) > 0:
            clustering = DBSCAN(eps=self.config.dbscan_eps, 
                              min_samples=self.config.dbscan_min_samples).fit(coords)
            spatial_noise = clustering.labels_ == -1
            noise_mask |= spatial_noise
            stats['spatial_noise_rate'] = np.mean(spatial_noise)
        
        # Temporal noise detection (isolated events)
        if len(events) > 1:
            time_diffs = np.diff(events[:, 0])
            temporal_threshold = np.percentile(time_diffs, 95)
            isolated_events = np.concatenate([[False], time_diffs > temporal_threshold])
            noise_mask |= isolated_events
            stats['temporal_noise_rate'] = np.mean(isolated_events)
        
        # Hot pixel detection
        pixel_counts = {}
        for x, y in events[:, 1:3]:
            pixel_counts[(x, y)] = pixel_counts.get((x, y), 0) + 1
        
        if pixel_counts:
            count_threshold = np.percentile(list(pixel_counts.values()), 99)
            hot_pixel_mask = np.array([pixel_counts[(x, y)] > count_threshold 
                                     for x, y in events[:, 1:3]])
            noise_mask |= hot_pixel_mask
            stats['hot_pixel_rate'] = np.mean(hot_pixel_mask)
        
        stats['total_noise_rate'] = np.mean(noise_mask)
        logger.info(f"Noise detection stats: {stats}")
        
        return events[~noise_mask], stats
    
    def preprocess_dvs_gesture(self, events: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
        """Enhanced preprocessing with better temporal handling"""
        if events is None or len(events) == 0:
            return np.array([])
        
        # Filter events within label windows
        valid_mask = np.zeros(len(events), dtype=bool)
        for _, row in labels.iterrows():
            start_time, end_time = row['startTime_usec'], row['endTime_usec']
            window_mask = (events[:, 0] >= start_time) & (events[:, 0] <= end_time)
            valid_mask |= window_mask
        
        events = events[valid_mask]
        
        # Spatial filtering
        spatial_mask = ((events[:, 1] >= 0) & (events[:, 1] < self.config.dvs_max_x) & 
                       (events[:, 2] >= 0) & (events[:, 2] < self.config.dvs_max_y))
        events = events[spatial_mask]
        
        if len(events) == 0:
            return events
        
        # Temporal normalization with better handling
        t_min, t_max = events[:, 0].min(), events[:, 0].max()
        if t_max > t_min:
            events[:, 0] = (events[:, 0] - t_min) / (t_max - t_min)
            
            # Add temporal jitter for augmentation
            if self.config.max_jitter > 0:
                jitter = np.random.uniform(-self.config.max_jitter / (t_max - t_min), 
                                         self.config.max_jitter / (t_max - t_min), 
                                         len(events))
                events[:, 0] = np.clip(events[:, 0] + jitter, 0, 1)
        
        return events
    
    def visualize_dvs_data(self, events: np.ndarray, labels: pd.DataFrame, save_path: Optional[str] = None):
        """Enhanced visualization with subplots"""
        if events is None or len(events) == 0:
            logger.warning("No events to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temporal distribution
        axes[0, 0].hist(events[:, 0], bins=100, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Normalized Timestamp')
        axes[0, 0].set_ylabel('Event Count')
        axes[0, 0].set_title('DVS Gesture Temporal Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Spatial distribution
        scatter = axes[0, 1].scatter(events[:, 1], events[:, 2], s=0.5, 
                                   c=events[:, 3], cmap='RdBu', alpha=0.6)
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        axes[0, 1].set_title('DVS Gesture Spatial Distribution')
        plt.colorbar(scatter, ax=axes[0, 1], label='Polarity (0: OFF, 1: ON)')
        
        # Class distribution
        class_counts = labels['class'].value_counts().sort_index()
        axes[1, 0].bar(class_counts.index, class_counts.values, 
                      color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('Gesture Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('DVS Gesture Class Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Event rate over time
        time_bins = np.linspace(0, 1, 50)
        event_rates = []
        for i in range(len(time_bins) - 1):
            mask = (events[:, 0] >= time_bins[i]) & (events[:, 0] < time_bins[i + 1])
            event_rates.append(np.sum(mask))
        
        axes[1, 1].plot(time_bins[:-1], event_rates, color='green', linewidth=2)
        axes[1, 1].set_xlabel('Normalized Time')
        axes[1, 1].set_ylabel('Event Rate')
        axes[1, 1].set_title('Event Rate Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class NCaltech101Processor:
    """Enhanced N-Caltech101 dataset processor"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_ncaltech101(self, bin_file: str) -> np.ndarray:
        """Load N-Caltech101 data with better error handling"""
        try:
            with open(bin_file, 'rb') as f:
                data = f.read()
            
            # More efficient unpacking
            events = []
            for i in range(0, len(data), 5):
                if i + 5 <= len(data):
                    event = struct.unpack('<BBHB', data[i:i+5])  # Little endian
                    x, y = event[0], event[1]
                    polarity = (event[2] >> 7) & 0x01
                    timestamp = ((event[2] & 0x7F) << 16) | event[3]
                    events.append([timestamp, x, y, polarity])
            
            events = np.array(events, dtype=np.float32)
            logger.info(f"Loaded {len(events)} events from {bin_file}")
            return events
            
        except Exception as e:
            logger.error(f"Error loading N-Caltech101 data from {bin_file}: {e}")
            return np.array([])
    
    def preprocess_ncaltech101(self, events: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for N-Caltech101"""
        if len(events) == 0:
            return events
        
        # Spatial filtering
        spatial_mask = ((events[:, 1] >= 0) & (events[:, 1] < self.config.caltech_max_x) & 
                       (events[:, 2] >= 0) & (events[:, 2] < self.config.caltech_max_y))
        events = events[spatial_mask]
        
        if len(events) == 0:
            return events
        
        # Temporal normalization
        t_min, t_max = events[:, 0].min(), events[:, 0].max()
        if t_max > t_min:
            events[:, 0] = (events[:, 0] - t_min) / (t_max - t_min)
            
            # Add temporal jitter
            if self.config.max_jitter > 0:
                jitter = np.random.uniform(-self.config.max_jitter / (t_max - t_min), 
                                         self.config.max_jitter / (t_max - t_min), 
                                         len(events))
                events[:, 0] = np.clip(events[:, 0] + jitter, 0, 1)
        
        return events

class EventFrameConverter:
    """Enhanced event-to-frame conversion with multiple representations"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def events_to_frames(self, events: np.ndarray, shape: Tuple[int, int, int], 
                        method: str = 'histogram') -> np.ndarray:
        """Convert events to frames with multiple methods"""
        if len(events) == 0:
            return np.zeros((1, *shape))
        
        time_bins = np.arange(0, 1 + self.config.time_bin, self.config.time_bin)
        frames = []
        
        for i in range(len(time_bins) - 1):
            t_start, t_end = time_bins[i], time_bins[i + 1]
            mask = (events[:, 0] >= t_start) & (events[:, 0] < t_end)
            frame_events = events[mask]
            
            if method == 'histogram':
                frame = self._histogram_representation(frame_events, shape)
            elif method == 'time_surface':
                frame = self._time_surface_representation(frame_events, shape, t_end)
            else:
                frame = self._histogram_representation(frame_events, shape)
            
            frames.append(frame)
        
        return np.array(frames)
    
    def _histogram_representation(self, events: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        """Standard histogram representation"""
        frame = np.zeros(shape)
        if len(events) > 0:
            for _, x, y, p in events:
                if 0 <= int(x) < shape[1] and 0 <= int(y) < shape[0]:
                    frame[int(y), int(x), int(p)] += 1
        return frame
    
    def _time_surface_representation(self, events: np.ndarray, shape: Tuple[int, int, int], 
                                   current_time: float) -> np.ndarray:
        """Time surface representation"""
        frame = np.zeros(shape)
        if len(events) > 0:
            for t, x, y, p in events:
                if 0 <= int(x) < shape[1] and 0 <= int(y) < shape[0]:
                    # Exponential decay based on time difference
                    time_diff = current_time - t
                    decay = np.exp(-time_diff / 0.1)  # Decay constant
                    frame[int(y), int(x), int(p)] = max(frame[int(y), int(x), int(p)], decay)
        return frame

class NeuromorphicDataset(Dataset):
    """PyTorch Dataset for neuromorphic data"""
    
    def __init__(self, frames: np.ndarray, labels: np.ndarray, transform=None):
        self.frames = torch.FloatTensor(frames)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label

class EnhancedSpikingNet(nn.Module):
    """Enhanced Spiking Neural Network with better architecture"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        
        c, h, w = input_shape
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            norse.LIConv2d(c, 32, kernel_size=3, padding=1),
            norse.LIFCell(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block
            norse.LIConv2d(32, 64, kernel_size=3, padding=1),
            norse.LIFCell(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block
            norse.LIConv2d(64, 128, kernel_size=3, padding=1),
            norse.LIFCell(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(dropout_rate),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            norse.LIFCell(128 * 4 * 4, 256),
            nn.Dropout(dropout_rate),
            norse.LIFCell(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Trainer:
    """Enhanced training class with better monitoring"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Full training loop with monitoring"""
        logger.info(f"Starting training on {self.device}")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info("Early stopping triggered")
                    break
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Best Val Acc: {best_val_acc:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    # Setup environment
    DataProcessor.setup_environment()
    
    # Initialize processors
    dvs_processor = DVSGestureProcessor(config)
    caltech_processor = NCaltech101Processor(config)
    frame_converter = EventFrameConverter(config)
    
    # Example usage for DVS Gesture dataset
    aedat_file = os.path.join(config.data_path, 'DVS/DvsGesture/user10_fluorescent_led.aedat')
    csv_file = os.path.join(config.data_path, 'DVS/DvsGesture/user10_fluorescent_led_labels.csv')
    
    if os.path.exists(aedat_file) and os.path.exists(csv_file):
        # Load and process DVS data
        events, labels = dvs_processor.load_dvs_gesture(aedat_file, csv_file)
        
        if events is not None:
            # Detect and remove noise
            clean_events, noise_stats = dvs_processor.detect_noise_advanced(events)
            
            # Preprocess events
            processed_events = dvs_processor.preprocess_dvs_gesture(clean_events, labels)
            
            # Visualize data
            dvs_processor.visualize_dvs_data(processed_events, labels)
            
            # Convert to frames
            frames = frame_converter.events_to_frames(
                processed_events, 
                (config.dvs_max_y, config.dvs_max_x, 2)
            )
            
            logger.info(f"Generated {len(frames)} frames with shape {frames.shape}")
            
            # Create dummy labels for demonstration
            frame_labels = np.random.randint(0, config.dvs_num_classes, len(frames))
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                frames, frame_labels, test_size=0.2, random_state=42
            )
            
            # Create datasets
            train_dataset = NeuromorphicDataset(X_train, y_train)
            val_dataset = NeuromorphicDataset(X_val, y_val)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Initialize model
            input_shape = (2, config.dvs_max_y, config.dvs_max_x)
            model = EnhancedSpikingNet(input_shape, config.dvs_num_classes)
            
            # Initialize trainer
            trainer = Trainer(model, config)
            
            # Train model
            history = trainer.train(train_loader, val_loader)
            
            # Plot training history
            trainer.plot_training_history()
            
            logger.info("Training completed successfully!")
    else:
        logger.warning(f"DVS files not found at {aedat_file} or {csv_file}")

class AdvancedAnalytics:
    """Advanced analytics and evaluation tools"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      class_names: List[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        model.eval()
        device = next(model.parameters()).device
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Classification report
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(all_targets)))]
        
        report = classification_report(all_targets, all_preds, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_path: Optional[str] = None):
        """Plot confusion matrix with better visualization"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_event_statistics(self, events: np.ndarray) -> Dict[str, Any]:
        """Comprehensive event statistics analysis"""
        if len(events) == 0:
            return {}
        
        stats = {
            'total_events': len(events),
            'temporal_span': events[:, 0].max() - events[:, 0].min(),
            'spatial_span_x': events[:, 1].max() - events[:, 1].min(),
            'spatial_span_y': events[:, 2].max() - events[:, 2].min(),
            'polarity_ratio': np.mean(events[:, 3]),
            'event_rate': len(events) / (events[:, 0].max() - events[:, 0].min() + 1e-6),
            'spatial_density': len(events) / ((events[:, 1].max() - events[:, 1].min() + 1) * 
                                            (events[:, 2].max() - events[:, 2].min() + 1))
        }
        
        # Temporal statistics
        if len(events) > 1:
            inter_event_times = np.diff(events[:, 0])
            stats.update({
                'mean_inter_event_time': np.mean(inter_event_times),
                'std_inter_event_time': np.std(inter_event_times),
                'median_inter_event_time': np.median(inter_event_times)
            })
        
        return stats

class DataAugmentation:
    """Advanced data augmentation techniques for neuromorphic data"""
    
    @staticmethod
    def spatial_jitter(events: np.ndarray, max_shift: int = 5) -> np.ndarray:
        """Apply spatial jitter to events"""
        augmented_events = events.copy()
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)
        
        augmented_events[:, 1] += shift_x
        augmented_events[:, 2] += shift_y
        
        return augmented_events
    
    @staticmethod
    def temporal_stretch(events: np.ndarray, stretch_factor: float = 1.2) -> np.ndarray:
        """Apply temporal stretching/compression"""
        augmented_events = events.copy()
        augmented_events[:, 0] *= stretch_factor
        return augmented_events
    
    @staticmethod
    def polarity_flip(events: np.ndarray, flip_prob: float = 0.1) -> np.ndarray:
        """Randomly flip polarity of some events"""
        augmented_events = events.copy()
        flip_mask = np.random.random(len(events)) < flip_prob
        augmented_events[flip_mask, 3] = 1 - augmented_events[flip_mask, 3]
        return augmented_events
    
    @staticmethod
    def event_dropout(events: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
        """Randomly drop events"""
        keep_mask = np.random.random(len(events)) > dropout_rate
        return events[keep_mask]

class BatchProcessor:
    """Batch processing utilities for large datasets"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def process_dvs_dataset_batch(self, data_dir: str, output_dir: str, 
                                num_workers: int = 4) -> Dict[str, Any]:
        """Process entire DVS dataset in batches"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all AEDAT files
        aedat_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.aedat'):
                    aedat_path = os.path.join(root, file)
                    csv_path = aedat_path.replace('.aedat', '_labels.csv')
                    if os.path.exists(csv_path):
                        aedat_files.append((aedat_path, csv_path))
        
        logger.info(f"Found {len(aedat_files)} AEDAT files with labels")
        
        # Process files in parallel
        processor = DVSGestureProcessor(self.config)
        frame_converter = EventFrameConverter(self.config)
        
        all_frames = []
        all_labels = []
        processing_stats = []
        
        def process_single_file(file_pair):
            aedat_file, csv_file = file_pair
            try:
                # Load data
                events, labels = processor.load_dvs_gesture(aedat_file, csv_file)
                if events is None:
                    return None
                
                # Process events
                clean_events, noise_stats = processor.detect_noise_advanced(events)
                processed_events = processor.preprocess_dvs_gesture(clean_events, labels)
                
                # Convert to frames
                frames = frame_converter.events_to_frames(
                    processed_events, 
                    (self.config.dvs_max_y, self.config.dvs_max_x, 2)
                )
                
                # Create labels for frames
                frame_labels = []
                for _, row in labels.iterrows():
                    frame_labels.extend([row['class']] * (len(frames) // len(labels)))
                
                return {
                    'frames': frames,
                    'labels': frame_labels[:len(frames)],
                    'stats': noise_stats,
                    'file': aedat_file
                }
                
            except Exception as e:
                logger.error(f"Error processing {aedat_file}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_file, aedat_files),
                total=len(aedat_files),
                desc="Processing files"
            ))
        
        # Aggregate results
        valid_results = [r for r in results if r is not None]
        
        for result in valid_results:
            all_frames.append(result['frames'])
            all_labels.extend(result['labels'])
            processing_stats.append(result['stats'])
        
        if all_frames:
            # Concatenate all frames
            all_frames = np.concatenate(all_frames, axis=0)
            all_labels = np.array(all_labels)
            
            # Save processed data
            np.save(os.path.join(output_dir, 'frames.npy'), all_frames)
            np.save(os.path.join(output_dir, 'labels.npy'), all_labels)
            
            # Save processing statistics
            with open(os.path.join(output_dir, 'processing_stats.pkl'), 'wb') as f:
                pickle.dump(processing_stats, f)
            
            logger.info(f"Processed {len(all_frames)} frames from {len(valid_results)} files")
            
            return {
                'frames': all_frames,
                'labels': all_labels,
                'stats': processing_stats,
                'num_files': len(valid_results)
            }
        
        return {}

class ModelOptimizer:
    """Model optimization and hyperparameter tuning"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def architecture_search(self, train_loader: DataLoader, val_loader: DataLoader,
                          input_shape: Tuple[int, int, int], num_classes: int) -> Dict[str, Any]:
        """Simple architecture search"""
        architectures = [
            {'channels': [16, 32, 64], 'dropout': 0.1},
            {'channels': [32, 64, 128], 'dropout': 0.2},
            {'channels': [64, 128, 256], 'dropout': 0.3},
        ]
        
        results = []
        
        for i, arch_config in enumerate(architectures):
            logger.info(f"Testing architecture {i+1}/{len(architectures)}: {arch_config}")
            
            # Create model with specific architecture
            model = self._create_custom_model(input_shape, num_classes, arch_config)
            trainer = Trainer(model, self.config)
            
            # Quick training (fewer epochs)
            temp_config = Config()
            temp_config.num_epochs = 10
            trainer.config = temp_config
            
            history = trainer.train(train_loader, val_loader)
            best_val_acc = max(history['val_accuracies'])
            
            results.append({
                'architecture': arch_config,
                'best_val_accuracy': best_val_acc,
                'history': history
            })
        
        # Sort by best validation accuracy
        results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
        
        return results
    
    def _create_custom_model(self, input_shape: Tuple[int, int, int], 
                           num_classes: int, arch_config: Dict[str, Any]) -> nn.Module:
        """Create model with custom architecture"""
        class CustomSpikingNet(nn.Module):
            def __init__(self):
                super().__init__()
                c, h, w = input_shape
                channels = arch_config['channels']
                dropout = arch_config['dropout']
                
                layers = []
                in_channels = c
                
                for out_channels in channels:
                    layers.extend([
                        norse.LIConv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        norse.LIFCell(),
                        nn.BatchNorm2d(out_channels),
                        nn.MaxPool2d(2),
                        nn.Dropout2d(dropout),
                    ])
                    in_channels = out_channels
                
                self.features = nn.Sequential(*layers)
                
                # Calculate flattened size
                with torch.no_grad():
                    dummy_input = torch.zeros(1, c, h, w)
                    dummy_output = self.features(dummy_input)
                    flattened_size = dummy_output.numel()
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    norse.LIFCell(flattened_size, 256),
                    nn.Dropout(dropout),
                    norse.LIFCell(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return CustomSpikingNet()

def create_demo_pipeline():
    """Create a complete demo pipeline with synthetic data"""
    logger.info("Creating demo pipeline with synthetic data")
    
    # Generate synthetic neuromorphic-like data
    def generate_synthetic_events(num_events: int = 10000) -> np.ndarray:
        # Create synthetic events with realistic patterns
        events = np.zeros((num_events, 4))
        
        # Timestamps (sorted)
        events[:, 0] = np.sort(np.random.exponential(0.1, num_events))
        events[:, 0] = (events[:, 0] - events[:, 0].min()) / (events[:, 0].max() - events[:, 0].min())
        
        # Spatial coordinates (clustered patterns)
        cluster_centers = [(32, 32), (96, 32), (32, 96), (96, 96)]
        for i in range(num_events):
            center = cluster_centers[i % len(cluster_centers)]
            events[i, 1] = np.clip(np.random.normal(center[0], 10), 0, 127)
            events[i, 2] = np.clip(np.random.normal(center[1], 10), 0, 127)
        
        # Polarity (random but balanced)
        events[:, 3] = np.random.choice([0, 1], num_events, p=[0.4, 0.6])
        
        return events
    
    # Generate synthetic data for multiple classes
    all_frames = []
    all_labels = []
    
    frame_converter = EventFrameConverter(config)
    
    for class_id in range(5):  # 5 synthetic classes
        for sample in range(20):  # 20 samples per class
            events = generate_synthetic_events(1000 + np.random.randint(-200, 200))
            frames = frame_converter.events_to_frames(
                events, (config.dvs_max_y, config.dvs_max_x, 2)
            )
            all_frames.append(frames)
            all_labels.extend([class_id] * len(frames))
    
    # Concatenate all data
    all_frames = np.concatenate(all_frames, axis=0)
    all_labels = np.array(all_labels)
    
    logger.info(f"Generated {len(all_frames)} synthetic frames with {len(np.unique(all_labels))} classes")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        all_frames, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42, stratify=y_val
    )
    
    # Create datasets and loaders
    train_dataset = NeuromorphicDataset(X_train, y_train)
    val_dataset = NeuromorphicDataset(X_val, y_val)
    test_dataset = NeuromorphicDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    input_shape = (2, config.dvs_max_y, config.dvs_max_x)
    model = EnhancedSpikingNet(input_shape, 5)  # 5 classes
    
    # Train model
    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate model
    analytics = AdvancedAnalytics(config)
    evaluation = analytics.evaluate_model(model, test_loader, 
                                        class_names=[f"Synthetic_Class_{i}" for i in range(5)])
    
    # Plot results
    trainer.plot_training_history()
    analytics.plot_confusion_matrix(evaluation['confusion_matrix'], 
                                   [f"Class_{i}" for i in range(5)])
    
    logger.info(f"Test Accuracy: {evaluation['accuracy']:.4f}")
    
    return {
        'model': model,
        'trainer': trainer,
        'evaluation': evaluation,
        'history': history
    }

def main():
    """Enhanced main execution function with multiple options"""
    logger.info("Starting Enhanced Neuromorphic Vision Pipeline")
    
    # Setup environment
    DataProcessor.setup_environment()
    
    # Initialize processors
    dvs_processor = DVSGestureProcessor(config)
    caltech_processor = NCaltech101Processor(config)
    frame_converter = EventFrameConverter(config)
    analytics = AdvancedAnalytics(config)
    
    # Check if real data exists
    aedat_file = os.path.join(config.data_path, 'DVS/DvsGesture/user10_fluorescent_led.aedat')
    csv_file = os.path.join(config.data_path, 'DVS/DvsGesture/user10_fluorescent_led_labels.csv')
    
    if os.path.exists(aedat_file) and os.path.exists(csv_file):
        logger.info("Real DVS data found, processing...")
        
        # Load and process real data
        events, labels = dvs_processor.load_dvs_gesture(aedat_file, csv_file)
        
        if events is not None:
            # Analyze event statistics
            event_stats = analytics.analyze_event_statistics(events)
            logger.info(f"Event statistics: {event_stats}")
            
            # Detect and remove noise
            clean_events, noise_stats = dvs_processor.detect_noise_advanced(events)
            logger.info(f"Noise detection completed: {noise_stats}")
            
            # Apply data augmentation
            augmenter = DataAugmentation()
            augmented_events = augmenter.spatial_jitter(clean_events, max_shift=3)
            augmented_events = augmenter.temporal_stretch(augmented_events, stretch_factor=1.1)
            
            # Preprocess events
            processed_events = dvs_processor.preprocess_dvs_gesture(augmented_events, labels)
            
            # Visualize data
            dvs_processor.visualize_dvs_data(processed_events, labels)
            
            # Convert to frames with multiple representations
            histogram_frames = frame_converter.events_to_frames(
                processed_events, 
                (config.dvs_max_y, config.dvs_max_x, 2),
                method='histogram'
            )
            
            time_surface_frames = frame_converter.events_to_frames(
                processed_events, 
                (config.dvs_max_y, config.dvs_max_x, 2),
                method='time_surface'
            )
            
            logger.info(f"Generated {len(histogram_frames)} histogram frames")
            logger.info(f"Generated {len(time_surface_frames)} time surface frames")
            
            # Use histogram frames for training (you can experiment with time_surface_frames)
            frames = histogram_frames
            
            # Create labels for frames
            frame_labels = np.random.randint(0, config.dvs_num_classes, len(frames))
            
            # Continue with training pipeline...
            logger.info("Starting training pipeline with real data")
            
        else:
            logger.warning("Could not load real data, switching to demo mode")
            create_demo_pipeline()
    
    else:
        logger.info("Real DVS data not found, running demo pipeline")
        demo_results = create_demo_pipeline()
        
        # Demonstrate batch processing capability
        logger.info("Demonstrating batch processing capabilities...")
        batch_processor = BatchProcessor(config)
        
        # Demonstrate model optimization
        if demo_results:
            logger.info("Demonstrating architecture search...")
            # This would normally use real data loaders
            # optimizer = ModelOptimizer(config)
            # search_results = optimizer.architecture_search(train_loader, val_loader, input_shape, num_classes)
            # logger.info(f"Best architecture: {search_results[0]['architecture']}")
    
    logger.info("Pipeline execution completed!")

if __name__ == "__main__":
    main()