from preprocessing import *

def train_model_indoor():
    print("Training indoor model...")
    indoor_dataset = IllumDataset('train')
    indoor_ids = [i for i in range(len(indoor_dataset)) if indoor_dataset[i][6] == 'indoor']
    indoor_datamodule = DataModule(indoor_dataset, val_size=VAL_SIZE, batch_size=BATCH_SIZE, ids=indoor_ids)
    model = HistPredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    ce_loss_fn = nn.CrossEntropyLoss()
    best_val_wass = float('inf')
    patience_counter = 0
    best_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(indoor_datamodule.get_train_dataloader(), desc=f"Indoor Epoch {epoch+1}"):
            img, chroma_hist, illum_hist, edge_map, depth, _, _, brightness, saturation, light_type, patch_stats, _ = batch
            img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, light_type, patch_stats = (
                img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                brightness.to(DEVICE), saturation.to(DEVICE), light_type.to(DEVICE), patch_stats.to(DEVICE))
            optimizer.zero_grad()
            pred_hist, _, _, _, _, light_type_logits = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
            illum_hist = illum_hist / (illum_hist.sum(dim=(1,2), keepdim=True) + 1e-10)
            loss_kl = kl_loss_fn(torch.log(pred_hist + 1e-10), illum_hist)
            loss_wass = sliced_wasserstein_loss(pred_hist, illum_hist)
            loss_light = ce_loss_fn(light_type_logits, light_type)
            loss = KL_WEIGHT * loss_kl + WASSERSTEIN_WEIGHT * loss_wass + LIGHT_TYPE_LOSS_WEIGHT * loss_light
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, pred_hist, light_type, patch_stats
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Indoor Epoch {epoch+1}, Train Loss: {train_loss / len(indoor_datamodule.get_train_dataloader()):.4f}")
        model.eval()
        val_wass = []
        with torch.no_grad():
            for batch in tqdm(indoor_datamodule.get_val_dataloader(), desc="Indoor Validation"):
                img, chroma_hist, illum_hist, edge_map, depth, _, _, brightness, saturation, _, patch_stats, _ = batch
                img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, patch_stats = (
                    img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                    brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
                pred_hist, _, _, _, _, _ = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
                for i in range(pred_hist.size(0)):
                    wass = sliced_wasserstein_loss(pred_hist[i:i+1], illum_hist[i:i+1]).item()
                    val_wass.append(wass)
                del img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, pred_hist, patch_stats
                gc.collect()
                torch.cuda.empty_cache()
        val_wass_mean = np.mean(val_wass)
        print(f"Indoor Val Wasserstein: {val_wass_mean:.4f}")
        scheduler.step()
        if val_wass_mean < best_val_wass:
            best_val_wass = val_wass_mean
            best_epoch = epoch + 1
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model_indoor.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Indoor Early stopping at epoch {epoch+1}. Best Val Wass: {best_val_wass:.4f} at epoch {best_epoch}")
                break
    print(f"Indoor Training completed. Best model saved to {OUTPUT_DIR / 'best_model_indoor.pth'}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_wass
def train_model_outdoor():
    print("Training outdoor model...")
    outdoor_dataset = IllumDataset('train')
    outdoor_ids = [i for i in range(len(outdoor_dataset)) if outdoor_dataset[i][6] == 'outdoor']
    outdoor_datamodule = DataModule(outdoor_dataset, val_size=VAL_SIZE, batch_size=BATCH_SIZE, ids=outdoor_ids)
    model = HistPredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    ce_loss_fn = nn.CrossEntropyLoss()
    best_val_wass = float('inf')
    patience_counter = 0
    best_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(outdoor_datamodule.get_train_dataloader(), desc=f"Outdoor Epoch {epoch+1}"):
            img, chroma_hist, illum_hist, edge_map, depth, _, _, brightness, saturation, light_type, patch_stats, _ = batch
            img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, light_type, patch_stats = (
                img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                brightness.to(DEVICE), saturation.to(DEVICE), light_type.to(DEVICE), patch_stats.to(DEVICE))
            optimizer.zero_grad()
            pred_hist, _, _, _, _, light_type_logits = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
            illum_hist = illum_hist / (illum_hist.sum(dim=(1,2), keepdim=True) + 1e-10)
            loss_kl = kl_loss_fn(torch.log(pred_hist + 1e-10), illum_hist)
            loss_wass = sliced_wasserstein_loss(pred_hist, illum_hist)
            loss_light = ce_loss_fn(light_type_logits, light_type)
            loss = KL_WEIGHT * loss_kl + WASSERSTEIN_WEIGHT * loss_wass + LIGHT_TYPE_LOSS_WEIGHT * loss_light
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, pred_hist, light_type, patch_stats
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Outdoor Epoch {epoch+1}, Train Loss: {train_loss / len(outdoor_datamodule.get_train_dataloader()):.4f}")
        model.eval()
        val_wass = []
        with torch.no_grad():
            for batch in tqdm(outdoor_datamodule.get_val_dataloader(), desc="Outdoor Validation"):
                img, chroma_hist, illum_hist, edge_map, depth, _, _, brightness, saturation, _, patch_stats, _ = batch
                img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, patch_stats = (
                    img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                    brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
                pred_hist, _, _, _, _, _ = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
                for i in range(pred_hist.size(0)):
                    wass = sliced_wasserstein_loss(pred_hist[i:i+1], illum_hist[i:i+1]).item()
                    val_wass.append(wass)
                del img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, pred_hist, patch_stats
                gc.collect()
                torch.cuda.empty_cache()
        val_wass_mean = np.mean(val_wass)
        print(f"Outdoor Val Wasserstein: {val_wass_mean:.4f}")
        scheduler.step()
        if val_wass_mean < best_val_wass:
            best_val_wass = val_wass_mean
            best_epoch = epoch + 1
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model_outdoor.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Outdoor Early stopping at epoch {epoch+1}. Best Val Wass: {best_val_wass:.4f} at epoch {best_epoch}")
                break
    print(f"Outdoor Training completed. Best model saved to {OUTPUT_DIR / 'best_model_outdoor.pth'}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_wass
def evaluate_and_visualize(use_light_value: bool = True, mode='indoor', lab_corr=False):
    initialize_midas()
    if mode == 'indoor':
        model_path = OUTPUT_DIR / 'best_model_indoor.pth'
        dataset = IllumDataset('train')
        ids = [i for i in range(len(dataset)) if dataset[i][6] == 'indoor']
    else:
        model_path = OUTPUT_DIR / 'best_model_outdoor.pth'
        dataset = IllumDataset('train')
        ids = [i for i in range(len(dataset)) if dataset[i][6] == 'outdoor']
    datamodule = DataModule(dataset, val_size=VAL_SIZE, batch_size=1, ids=ids)
    model = HistPredictor().to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Error: {model_path} not found")
        return
    model.eval()
    wass_values = []
    vis_count = 57
    global read_image
    original_read_image = read_image
    read_image = lambda path: original_read_image(path, white_level_corr=True, darken=True, use_light_value=use_light_value, lab_correction=lab_corr)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(datamodule.get_val_dataloader(), desc=f"{mode.capitalize()} Evaluation")):
            img, chroma_hist, illum_hist, edge_map, depth, path, _, brightness, saturation, _, patch_stats, _ = batch
            img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, patch_stats = (
                img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
            pred_hist, _, _, _, _, _ = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
            pred_np = pred_hist[0].cpu().numpy()
            gt_np = illum_hist[0].cpu().numpy()
            wass = sliced_wasserstein_loss(pred_hist, illum_hist).item()
            wass_values.append(wass)
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.title("Input Image")
            img_np = img[0].cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            plt.imshow(img_np)
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.title(f"Predicted Histogram (Wass: {wass:.3f})")
            plt.imshow(pred_np, cmap='hot')
            plt.colorbar()
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.title("Ground Truth Histogram")
            plt.imshow(gt_np, cmap='hot')
            plt.colorbar()
            plt.axis('off')
            plt.suptitle(f"Image: {Path(path[0]).name}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'{mode}_eval_example_{i+1}.png', dpi=300)
            plt.show()
            plt.close()
            if i >= vis_count - 1:
                break
    read_image = original_read_image
    mean_wass = np.mean(wass_values)
    std_wass = np.std(wass_values)
    print("Evaluation Metrics")
    print(f"Mean Val Wasserstein: {mean_wass:.3f}")
    print(f"Std Val Wasserstein: {std_wass:.3f}")
    with open(OUTPUT_DIR / f'{mode}_eval_metrics_light_value_{use_light_value}.txt', 'w') as f:
        f.write(f"Mean Val Wasserstein: {mean_wass:.3f}\n")
        f.write(f"Std Val Wasserstein: {std_wass:.3f}\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return mean_wass, std_wass

if __name__ == "__main__":
    initialize_midas()
    train_model_indoor()
    train_model_outdoor()
    print("Evaluating indoor...")
    evaluate_and_visualize(mode='indoor')
    print("Evaluating outdoor...")
    evaluate_and_visualize(mode='outdoor')
    print("=== LAB CORRECTION VALIDATION INDOOR ===")
    # Коррекци с включенным параметром lab_corr=True, то есть применяется коррекция к изображению в read_image(если включено)
    evaluate_and_visualize(mode='indoor', lab_corr=True)
    print("=== LAB CORRECTION VALIDATION OUTOOR ===")
    evaluate_and_visualize(mode='outdoor', lab_corr=True)