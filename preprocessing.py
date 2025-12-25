import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt
from typing import Literal
import random
import timm
from PIL import Image
import ot
import zipfile
from scipy.stats import entropy
import gc
import time
import json
from lime import lime_image

DATA_DIR = Path('')
TRAIN_IMGS_DIR = DATA_DIR / 'train_imgs' / 'train_imgs'
TRAIN_HISTS_DIR = DATA_DIR / 'train_histograms' / 'train_histograms'
TEST_IMGS_DIR = DATA_DIR / 'test_imgs' / 'test_imgs'
TRAIN_MARKUP_DIR = Path("train_content_markup/train_content_markup")
TEST_MARKUP_DIR = Path("test_content_markup/test_content_markup")
OUTPUT_DIR = Path('')
HISTOGRAMS_DIR = OUTPUT_DIR / 'histograms'
HISTOGRAMS_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 4
EPOCHS = 50
LR = 3e-5
VAL_SIZE = 0.1
PATIENCE = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WHITE_LEVEL = 2**16 - 1 - 256
HIST_BINS = [128, 128]
HIST_RANGE_FLAT = [-2, 2, -2, 2]
GAMMA = 1.0
NUM_SLICED_PROJECTIONS = 100
WASSERSTEIN_WEIGHT = 0.7
KL_WEIGHT = 0.3
NUM_SAMPLES_FOR_WASS = 1000
NUM_MIXTURES = 16
MIXUP_PROB = 0.5
NUM_PATCHES = 16 # Для local patch statistics, e.g., 4x4 grid
LIGHT_TYPES = ['sunny', 'cloudy', 'sunset', 'night', 'soft_light', 'low_light', 'multi_illuminant', 'sun'] # Список возможных light_type
NUM_LIGHT_CLASSES = len(LIGHT_TYPES)
LIGHT_TYPE_LOSS_WEIGHT = 0.1

metadata_df = pd.read_csv(DATA_DIR / 'metadata.csv')
metadata_dict = {row['names']: {'exposure_time': row['ExposureTime'], 'iso': row['ISO'], 'light_value': row['LightValue']}
                 for _, row in metadata_df.iterrows()}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

MIDAS_MODEL = None
def initialize_midas():
    global MIDAS_MODEL
    if MIDAS_MODEL is None:
        MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", force_reload=False).to(DEVICE).eval()

def apply_color_temperature(img: np.ndarray, temp: float) -> np.ndarray:
    temp_profiles = {
        2850: np.array([1.0, 0.8, 0.6]),
        5500: np.array([1.0, 1.0, 1.0]),
        6500: np.array([0.9, 1.0, 1.1])
    }
    profile = temp_profiles.get(temp, temp_profiles[5500])
    img_tinted = img * profile
    return np.clip(img_tinted, 0, 1)

def read_image(
    path: Path | str,
    white_level_corr: bool = True,
    darken: bool = True,
    use_light_value: bool = True,
    lab_correction: bool = False  
):
    path_obj = Path(path)
    name = f"{path_obj.parent.name}/{path_obj.name}"
    meta = metadata_dict.get(name, {'light_value': 10.0})

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")

    if white_level_corr:
        img = img / WHITE_LEVEL
    if use_light_value:
        img = img * meta['light_value']
    if darken:
        img = img * GAMMA
    # ----- МЕТОД ЦВЕТОВОЙ КОРРЕКЦИИ ---- 
    if lab_correction and img.shape[-1] == 3:
        #примеры и обьяснение в презентации 
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)

        # CLAHE — адаптивное выравнивание гистограммы для L-канала 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_clahe = clahe.apply(L)

        lab_clahe = cv2.merge((L_clahe, A, B))
        img_eq = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        img = img_eq.astype(np.float32) / 255.0

    # --- для MiDaS ---
    if img.shape[-1] == 3:
        img = img[..., ::-1].copy()
    img_tensor = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = MIDAS_MODEL(img_tensor)
        depth = F.interpolate(depth.unsqueeze(1), size=(224, 224), mode='bilinear').squeeze()

    # --- HSV анализ ---
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    brightness = np.mean(img, axis=2).astype(np.float32)

    return img.astype(np.float32), depth.cpu().numpy().astype(np.float32), brightness, saturation
def read_hist(path2hist: Path):
    hist = cv2.imread(str(path2hist), cv2.IMREAD_UNCHANGED)
    if hist is None:
        raise ValueError(f"Не удалось загрузить гистограмму: {path2hist}")
    hist = hist[6:-6, 14:-14]
    hist = cv2.resize(hist, (128, 128), interpolation=cv2.INTER_LINEAR)
    return (hist.astype(np.float32) / 255).astype(np.float32)
def save_hist(hist: np.ndarray, path: Path):
    hist = hist / (hist.sum() + 1e-10)
    hist = np.clip(hist / (hist.max() + 1e-10) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), hist)
def rgb2logchrom(rgb: np.ndarray):
    rgb = np.clip(rgb.astype(np.float32), 1e-6, None)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    u = np.log(r / g)
    v = np.log(b / g)
    u = np.where(np.isinf(u) | np.isnan(u), 0, u)
    v = np.where(np.isinf(v) | np.isnan(v), 0, v)
    return np.stack((u, v), axis=-1)
def generate_chroma_hist(img: np.ndarray, dropout_prob: float = 0.0):
    rgb = img.reshape(-1, 3)
    logchrom = rgb2logchrom(rgb)
    u, v = logchrom[:, 0], logchrom[:, 1]
    hist, _, _ = np.histogram2d(
        u, v, bins=HIST_BINS, range=[[HIST_RANGE_FLAT[0], HIST_RANGE_FLAT[1]], [HIST_RANGE_FLAT[2], HIST_RANGE_FLAT[3]]])
    hist = hist / (hist.sum() + 1e-10)
    if np.random.rand() < dropout_prob:
        mask = np.random.rand(*hist.shape) > 0.3
        hist = hist * mask
        hist = hist / (hist.sum() + 1e-10)
    return hist.astype(np.float32)
def get_edge_map(img: np.ndarray):
    gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return (edges / 255.0).astype(np.float32)
def sliced_wasserstein_loss(pred_hist: torch.Tensor, gt_hist: torch.Tensor):
    batch_size = pred_hist.size(0)
    loss = 0.0
    for i in range(batch_size):
        pred = pred_hist[i].flatten().detach().cpu().numpy()
        gt = gt_hist[i].flatten().detach().cpu().numpy()
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        if pred_sum <= 0:
            pred = np.ones_like(pred) / len(pred)
        else:
            pred = pred / pred_sum
        if gt_sum <= 0:
            gt = np.ones_like(gt) / len(gt)
        else:
            gt = gt / gt_sum
        if np.any(np.isnan(pred)):
            pred = np.ones_like(pred) / len(pred)
        if np.any(np.isnan(gt)):
            gt = np.ones_like(gt) / len(gt)
        alphas = np.linspace(HIST_RANGE_FLAT[2], HIST_RANGE_FLAT[3], HIST_BINS[1])
        betas = np.linspace(HIST_RANGE_FLAT[0], HIST_RANGE_FLAT[1], HIST_BINS[0])
        alphas, betas = np.meshgrid(alphas, betas)
        points = np.stack((alphas.flatten(), betas.flatten()), axis=-1)
        pred_samples = points[np.random.choice(len(points), size=NUM_SAMPLES_FOR_WASS, p=pred)]
        gt_samples = points[np.random.choice(len(points), size=NUM_SAMPLES_FOR_WASS, p=gt)]
        pred_samples_t = torch.tensor(pred_samples, device=DEVICE, dtype=torch.float32)
        gt_samples_t = torch.tensor(gt_samples, device=DEVICE, dtype=torch.float32)
        dim = pred_samples_t.size(1)
        thetas = torch.randn(NUM_SLICED_PROJECTIONS, dim, device=DEVICE, dtype=torch.float32)
        thetas = thetas / torch.norm(thetas, dim=1, keepdim=True)
        proj_pred = torch.mm(pred_samples_t, thetas.t())
        proj_gt = torch.mm(gt_samples_t, thetas.t())
        proj_pred, _ = torch.sort(proj_pred, dim=0)
        proj_gt, _ = torch.sort(proj_gt, dim=0)
        slice_loss = torch.mean(torch.abs(proj_pred - proj_gt))
        loss += slice_loss
    return loss / batch_size
def wasserstein_2d(pred_hist: np.ndarray, gt_hist: np.ndarray):
    pred = pred_hist.flatten()
    gt = gt_hist.flatten()
    pred = pred / (pred.sum() + 1e-10)
    gt = gt / (gt.sum() + 1e-10)
    alphas = np.linspace(HIST_RANGE_FLAT[2], HIST_RANGE_FLAT[3], HIST_BINS[1])
    betas = np.linspace(HIST_RANGE_FLAT[0], HIST_RANGE_FLAT[1], HIST_BINS[0])
    alphas, betas = np.meshgrid(alphas, betas)
    points = np.stack((alphas.flatten(), betas.flatten()), axis=1)
    cost_matrix = ot.dist(points, points, metric='euclidean') ** 2
    return ot.emd2(pred, gt, cost_matrix, numItermax=1000000)

class IllumDataset:
    def __init__(self, part: Literal['train', 'test'] = 'train'):
        self.part = part
        self.markup = self._load_markup()
        self._init_paths()
    def _load_markup(self):
        markup_dir = TRAIN_MARKUP_DIR if self.part == 'train' else TEST_MARKUP_DIR
        markup = {}
        for json_path in markup_dir.glob('*.json'):
            with open(json_path, 'r') as f:
                data = json.load(f)
            img_name = json_path.stem + '.png'
            markup[img_name] = {
                'indoor_outdoor': data['OutdoorIndoor'].lower(),
                'light_type': data.get('light_type', 'unknown') 
            }
        return markup
    def _init_paths(self):
        if self.part == 'train':
            self.imgs_paths = sorted(TRAIN_IMGS_DIR.glob('*.png'))
            self.hists_paths = sorted(TRAIN_HISTS_DIR.glob('*.png'))
            print(f"Train images: {len(self.imgs_paths)}, histograms: {len(self.hists_paths)}")
            if len(self.imgs_paths) != len(self.hists_paths):
                raise ValueError("Несоответствие изображений и гистограмм")
        else:
            self.imgs_paths = sorted(TEST_IMGS_DIR.glob('*.png'))
            self.hists_paths = [None] * len(self.imgs_paths)
            print(f"Test images: {len(self.imgs_paths)}")
    def __getitem__(self, idx):
        img_path = self.imgs_paths[idx]
        img_name = img_path.name
        markup_data = self.markup.get(img_name, {'indoor_outdoor': 'outdoor', 'light_type': 'unknown'})
        indoor_outdoor = markup_data['indoor_outdoor']
        light_type = markup_data['light_type']
        img, depth, brightness, saturation = read_image(img_path)
        chroma_hist = generate_chroma_hist(img, dropout_prob=0.1 if self.part == 'train' else 0.0)
        illum_hist = read_hist(self.hists_paths[idx]) if self.hists_paths[idx] else np.zeros((HIST_BINS[1], HIST_BINS[0]), dtype=np.float32)
        edge_map = get_edge_map(img)
       
        # Local Patch Statistics
        patch_size = int(sqrt(NUM_PATCHES))
        h, w, _ = img.shape
        patch_h, patch_w = h // patch_size, w // patch_size
        patches = [img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] for i in range(patch_size) for j in range(patch_size)]
        patch_stats = []
        for p in patches:
            mean = np.mean(p, axis=(0,1))
            var = np.var(p, axis=(0,1))
            skew = entropy(p.reshape(-1, 3) + 1e-10, axis=0) 
            patch_stats.append(np.concatenate([mean, var, skew]))
        patch_stats = np.array(patch_stats).flatten().astype(np.float32)
       

        meta = metadata_dict.get(f"{img_path.parent.name}/{img_name}", {'light_value': 10.0})
       
        return img, chroma_hist, illum_hist, edge_map, depth, str(img_path), indoor_outdoor, brightness, saturation, light_type, patch_stats, meta['light_value']
    def __len__(self):
        return len(self.imgs_paths)
class AWBDatasetWithMeta(Dataset):
    def __init__(self, dataset: IllumDataset, ids: list, require_transform: bool = False):
        self.dataset = dataset
        self.ids = ids
        self.is_test = dataset.part == 'test'
        self.common_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.edge_transforms = transforms.Compose([transforms.ToTensor()])
        self.depth_transforms = transforms.Compose([transforms.ToTensor()])
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5)),
        ])
        self.require_transform = require_transform
        self.light_type_to_idx = {lt: i for i, lt in enumerate(LIGHT_TYPES)}
    def __getitem__(self, idx):
        real_idx = self.ids[idx]
        img, chroma_hist, illum_hist, edge_map, depth, path, indoor_outdoor, brightness, saturation, light_type, patch_stats, light_value = self.dataset[real_idx]
       
        if self.require_transform and not self.is_test and np.random.rand() < MIXUP_PROB:
            second_idx = random.choice(self.ids)
            img2, chroma_hist2, illum_hist2, edge_map2, depth2, _, _, brightness2, saturation2, light_type2, patch_stats2, light_value2 = self.dataset[second_idx]
            temps = [2850, 5500, 6500]
            temp1, temp2 = random.sample(temps, 2)
            img1_tinted = apply_color_temperature(img, temp1)
            img2_tinted = apply_color_temperature(img2, temp2)
            alpha = np.random.beta(0.2, 0.2)
            mixed_img = alpha * img1_tinted + (1 - alpha) * img2_tinted
            mixed_chroma_hist = alpha * chroma_hist + (1 - alpha) * chroma_hist2
            mixed_illum_hist = alpha * illum_hist + (1 - alpha) * illum_hist2
            mixed_edge_map = alpha * edge_map + (1 - alpha) * edge_map2
            mixed_depth = alpha * depth + (1 - alpha) * depth2
            mixed_brightness = alpha * brightness + (1 - alpha) * brightness2
            mixed_saturation = alpha * saturation + (1 - alpha) * saturation2
            mixed_patch_stats = alpha * patch_stats + (1 - alpha) * patch_stats2
            mixed_light_type = light_type if alpha > 0.5 else light_type2 
            mixed_light_value = alpha * light_value + (1 - alpha) * light_value2
            mixed_path = path
        else:
            mixed_img = img
            mixed_chroma_hist = chroma_hist
            mixed_illum_hist = illum_hist
            mixed_edge_map = edge_map
            mixed_depth = depth
            mixed_brightness = brightness
            mixed_saturation = saturation
            mixed_patch_stats = patch_stats
            mixed_light_type = light_type
            mixed_light_value = light_value
            mixed_path = path
        mixed_img = (mixed_img * 255).clip(0, 255).astype(np.uint8)
        mixed_img = Image.fromarray(mixed_img)
        if self.require_transform:
            mixed_img = self.train_transforms(mixed_img)
        mixed_img = self.common_transforms(mixed_img)
        mixed_edge_map = cv2.resize(mixed_edge_map, (224, 224), interpolation=cv2.INTER_LINEAR)
        mixed_edge_map = np.clip(mixed_edge_map, 0, 1)
        mixed_edge_map = self.edge_transforms(Image.fromarray((mixed_edge_map * 255).astype(np.uint8)))
        mixed_depth = cv2.resize(mixed_depth, (224, 224), interpolation=cv2.INTER_LINEAR)
        mixed_depth = self.depth_transforms(Image.fromarray(mixed_depth))
        mixed_brightness = cv2.resize(mixed_brightness, (224, 224), interpolation=cv2.INTER_LINEAR)
        mixed_brightness = np.clip(mixed_brightness, 0, mixed_brightness.max())
        mixed_brightness = self.edge_transforms(Image.fromarray((mixed_brightness * 255 / (mixed_brightness.max() + 1e-6)).astype(np.uint8)))
        mixed_saturation = cv2.resize(mixed_saturation, (224, 224), interpolation=cv2.INTER_LINEAR)
        mixed_saturation = self.edge_transforms(Image.fromarray((mixed_saturation * 255).astype(np.uint8)))
       
        light_type_idx = self.light_type_to_idx.get(mixed_light_type, 0) 
       
        return (mixed_img, torch.tensor(mixed_chroma_hist), torch.tensor(mixed_illum_hist),
                mixed_edge_map, mixed_depth, mixed_path, indoor_outdoor, mixed_brightness, mixed_saturation,
                torch.tensor(light_type_idx), torch.tensor(mixed_patch_stats), mixed_light_value)
    def __len__(self):
        return len(self.ids)
class DataModule:
    def __init__(self, dataset: IllumDataset, val_size: float = 0.2, batch_size: int = 8, ids=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ids = ids if ids is not None else list(range(len(dataset)))
        self._make_train_val_split(val_size)
        self._init_dataloaders()
    def _make_train_val_split(self, val_size: float):
        n = len(self.ids)
        if val_size == 0.0:
            self.train_ids = self.ids
            self.val_ids = []
        else:
            permuted_ids = np.random.permutation(self.ids)
            self.val_ids = permuted_ids[:int(n * val_size)].tolist()
            self.train_ids = permuted_ids[int(n * val_size):].tolist()
    def _init_dataloaders(self):
        self.train_dataset = AWBDatasetWithMeta(self.dataset, self.train_ids, require_transform=self.dataset.part == 'train')
        self.val_dataset = AWBDatasetWithMeta(self.dataset, self.val_ids, require_transform=False)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    def get_train_dataloader(self) -> DataLoader:
        return self.train_dataloader
    def get_val_dataloader(self) -> DataLoader:
        return self.val_dataloader
class HistPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_huge_patch14_224', pretrained=True, num_classes=0)
        self.feature_dim = 1280
        self.chroma_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 128)
        )
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64)
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64)
        )
        self.bright_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64)
        )
        self.sat_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64)
        )

        self.patch_encoder = nn.Sequential(
            nn.Linear(NUM_PATCHES * 9, 256), nn.ReLU(), # 9 = 3(mean) + 3(var) + 3(skew)
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
        combined_dim = self.feature_dim + 128 + 64 + 64 + 64 + 64 + 64 
        self.fc_mu = nn.Linear(combined_dim, NUM_MIXTURES * 2)
        self.fc_sigma = nn.Linear(combined_dim, NUM_MIXTURES * 2)
        self.fc_cov12 = nn.Linear(combined_dim, NUM_MIXTURES)
        self.fc_pi = nn.Linear(combined_dim, NUM_MIXTURES)

        # ВЕКТОРИЗАЦИЯ, ТО ЕСТЬ ДОПОЛНИТЕЛЬНАЯ ЗАДАЧА МОДЕЛИ ПРЕДСКАЗАТЬ light_type, если угадала верно получает вознаграждение
        # OUTDOOR/INDOOR используется для стратификации, то есть одна модель обучается на фото indoor, другая на outdoor соответсвенно предсказывает только на том же типе данных, на котором обучалась
        # light_value - используется в препроцессинге фото функция read_image
        # То есть, грубо говоря модель не использует метаданных как таковых для обуения, НО:
        # она использует их для препроцессинга, стратификации и мультитаскинга 
        #обратите внимание в презентации 
        self.fc_light_type = nn.Linear(combined_dim, NUM_LIGHT_CLASSES)
    def forward(self, x, chroma_hist, edge_map, depth, brightness, saturation, patch_stats):
        features = self.backbone(x)
        chroma_features = self.chroma_encoder(chroma_hist.unsqueeze(1))
        edge_features = self.edge_encoder(edge_map)
        depth_features = self.depth_encoder(depth)
        bright_features = self.bright_encoder(brightness)
        sat_features = self.sat_encoder(saturation)
        patch_features = self.patch_encoder(patch_stats)
        combined = torch.cat((features, chroma_features, edge_features, depth_features, bright_features, sat_features, patch_features), dim=1)
        mu = self.fc_mu(combined).view(-1, NUM_MIXTURES, 2)
        sigma_raw = self.fc_sigma(combined).view(-1, NUM_MIXTURES, 2)
        cov12 = self.fc_cov12(combined).view(-1, NUM_MIXTURES)
        sigma = F.softplus(sigma_raw)
        pi = F.softmax(self.fc_pi(combined), dim=1)
        pred_hist = gmm_to_hist(mu, sigma, cov12, pi)
        light_type_logits = self.fc_light_type(combined)
        return pred_hist, mu, sigma, cov12, pi, light_type_logits
def gmm_to_hist(mu: torch.Tensor, sigma: torch.Tensor, cov12: torch.Tensor, pi: torch.Tensor, bins=HIST_BINS, range_flat=HIST_RANGE_FLAT):
    # ПАРАМЕТРИЗАЦИЯ ГИСТОГРАММЫ С ПОМОЩЬЮ мат модели GMM 
    batch_size = mu.size(0)
    hist = torch.zeros(batch_size, bins[0], bins[1], device=mu.device)
    u = torch.linspace(range_flat[0], range_flat[1], bins[0], device=mu.device)
    v = torch.linspace(range_flat[2], range_flat[3], bins[1], device=mu.device)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    grid = torch.stack((uu.flatten(), vv.flatten()), dim=-1)
   
    min_sigma = 1e-3
    sigma = torch.clamp(sigma, min=min_sigma)
   
    for b in range(batch_size):
        for k in range(NUM_MIXTURES):
            mean = mu[b, k]
            s_u = sigma[b, k, 0]
            s_v = sigma[b, k, 1]
            off = cov12[b, k]
           
            cov = torch.tensor([[s_u**2, off], [off, s_v**2]], device=mu.device)
            cov = cov + torch.eye(2, device=mu.device) * 1e-6 # jitter
           
            det = torch.det(cov)
            if det <= 0 or torch.isnan(det):
                cov = torch.diag(torch.clamp(sigma[b, k], min=min_sigma)**2)
                det = torch.det(cov)
           
            inv_cov = torch.inverse(cov)
            diff = grid - mean
            mahalanobis = torch.sum(diff @ inv_cov * diff, dim=1)
            exp_term = torch.exp(-0.5 * mahalanobis)
            pdf = pi[b, k] * exp_term / (2 * np.pi * torch.sqrt(det + 1e-6))
            hist[b] += pdf.view(bins[0], bins[1])
   
    hist_sum = hist.sum(dim=(1,2), keepdim=True)
    hist_sum = torch.where(hist_sum > 0, hist_sum, torch.ones_like(hist_sum))
    hist = hist / hist_sum
    return hist