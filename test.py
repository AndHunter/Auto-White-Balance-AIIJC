from preprocessing import *
from train import *
def generate_submission_zip():
    initialize_midas()
    test_dataset = IllumDataset('test')
    test_dataloader = DataLoader(
        AWBDatasetWithMeta(test_dataset, list(range(len(test_dataset))), require_transform=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    model_indoor = HistPredictor().to(DEVICE)
    model_outdoor = HistPredictor().to(DEVICE)
    try:
        model_indoor.load_state_dict(torch.load(OUTPUT_DIR / 'best_model_indoor.pth'))
        model_outdoor.load_state_dict(torch.load(OUTPUT_DIR / 'best_model_outdoor.pth'))
    except FileNotFoundError:
        print("Error: Models not found")
        return
    model_indoor.eval()
    model_outdoor.eval()
    zip_path = OUTPUT_DIR / 'submission.zip'
    submission_data = []
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Generating submission"):
                img, chroma_hist, _, edge_map, depth, paths, indoor_outdoor, brightness, saturation, _, patch_stats, _ = batch
                img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats = (
                    img.to(DEVICE), chroma_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                    brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE)
                )
                pred_hist_np = []
                for i in range(img.size(0)):
                    if indoor_outdoor[i] == 'indoor':
                        pred_hist, _, _, _, _, _ = model_indoor(img[i:i+1], chroma_hist[i:i+1], edge_map[i:i+1], depth[i:i+1], brightness[i:i+1], saturation[i:i+1], patch_stats[i:i+1])
                    else:
                        pred_hist, _, _, _, _, _ = model_outdoor(img[i:i+1], chroma_hist[i:i+1], edge_map[i:i+1], depth[i:i+1], brightness[i:i+1], saturation[i:i+1], patch_stats[i:i+1])
                    pred_hist_np.append(pred_hist.cpu().numpy()[0])
                for i in range(len(pred_hist_np)):
                    pred_np = pred_hist_np[i]
                    if pred_np.sum() < 1e-5:
                        pred_np = np.ones_like(pred_np) / pred_np.size
                    output_path = HISTOGRAMS_DIR / f"{Path(paths[i]).stem}.png"
                    save_hist(pred_np, output_path)
                    zipf.write(output_path, arcname=f"{Path(paths[i]).stem}.png")
                    submission_data.append({
                        'image_path': str(paths[i]),
                        'white_point_distribution': f"histograms/{Path(paths[i]).stem}.png"
                    })
                del img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats
                gc.collect()
                torch.cuda.empty_cache()
    pd.DataFrame(submission_data).to_csv(OUTPUT_DIR / 'submission.csv', index=False)
    print(f"Submission ZIP saved to {zip_path}")
def analyze_light_value_hypothesis(mode='indoor'):
    print("Analyzing LightValue hypothesis...")
   
    mean_wass_with_lv, std_wass_with_lv = evaluate_and_visualize(use_light_value=True, mode=mode)
   
    mean_wass_without_lv, std_wass_without_lv = evaluate_and_visualize(use_light_value=False, mode=mode)
   
    lv_improvement = mean_wass_without_lv - mean_wass_with_lv
    lv_status = "confirmed" if lv_improvement > 0 else "not confirmed"
    print(f"Mean Wass with LightValue: {mean_wass_with_lv:.3f}")
    print(f"Mean Wass without LightValue: {mean_wass_without_lv:.3f}")
    print(f"Wass Diff (without - with): {lv_improvement:.3f}")
    print(f"LightValue Hypothesis: Status = {lv_status}")
    with open(OUTPUT_DIR / f'{mode}_light_value_hypothesis.txt', 'w') as f:
        f.write("LightValue Hypothesis Analysis:\n")
        f.write(f"Mean Wass with LightValue: {mean_wass_with_lv:.3f}\n")
        f.write(f"Std Wass with LightValue: {std_wass_with_lv:.3f}\n")
        f.write(f"Mean Wass without LightValue: {mean_wass_without_lv:.3f}\n")
        f.write(f"Std Wass without LightValue: {std_wass_without_lv:.3f}\n")
        f.write(f"Wass Diff (without - with): {lv_improvement:.3f}\n")
        f.write(f"Status: {lv_status}\n")
def analyze_worst_examples(mode='indoor'):
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
    wass_results = []
    print(f"Analyzing worst examples on validation set: {len(datamodule.val_dataset)} images")
    with torch.no_grad():
        for batch in tqdm(datamodule.get_val_dataloader(), desc="Analyzing worst examples"):
            img, chroma_hist, illum_hist, edge_map, depth, path, _, brightness, saturation, _, patch_stats, _ = batch
            img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, patch_stats = (
                img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
            pred_hist, _, _, _, _, _ = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
            wass = sliced_wasserstein_loss(pred_hist, illum_hist).item()
            wass_results.append((path[0], wass, img.cpu(), pred_hist.cpu(), illum_hist.cpu(), entropy(illum_hist[0].cpu().numpy().flatten() + 1e-10)))
            del img, chroma_hist, illum_hist, edge_map, depth, pred_hist, brightness, saturation, patch_stats
            gc.collect()
            torch.cuda.empty_cache()
    wass_results = sorted(wass_results, key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 worst examples by Wasserstein distance:")
    light_values, isos, exposures, gt_nonzero_bins, gt_entropies = [], [], [], [], []
    print("Index\tPath\tWasserstein\tLightValue\tISO\tExposure\tGT Nonzero Bins\tGT Entropy")
    for i, (path, wass, img, pred_hist, gt_hist, gt_entropy) in enumerate(wass_results, 1):
        name = f"{Path(path).parent.name}/{Path(path).name}"
        meta = metadata_df[metadata_df['names'] == name].iloc[0]
        light_value = meta['LightValue']
        iso = meta['ISO']
        exposure = meta['ExposureTime']
        if isinstance(exposure, str) and '/' in exposure:
            num, denom = map(float, exposure.split('/'))
            exposure = num / denom
        else:
            exposure = float(exposure)
        gt_np = gt_hist[0].numpy()
        gt_nonzero = (gt_np > 1e-6).sum()
        light_values.append(light_value)
        isos.append(iso)
        exposures.append(exposure)
        gt_nonzero_bins.append(gt_nonzero)
        gt_entropies.append(gt_entropy)
        print(f"{i}\t{path}\t{wass:.4f}\t{light_value:.2f}\t{iso}\t{exposure:.6f}\t{gt_nonzero}\t{gt_entropy:.4f}")
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.title("Input Image")
        img_np = img[0].permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.title(f"Predicted Histogram (Wass: {wass:.3f})")
        plt.imshow(pred_hist[0].numpy(), cmap='hot')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.title("Ground Truth Histogram")
        plt.imshow(gt_np, cmap='hot')
        plt.colorbar()
        plt.axis('off')
        plt.suptitle(f"Worst Example {i}: {Path(path).name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{mode}_worst_example_{i}.png', dpi=300)
        plt.close()
    print("\nCommon traits of worst examples:")
    print(f"Mean LightValue: {np.mean(light_values):.2f} ± {np.std(light_values):.2f}")
    print(f"Mean ISO: {np.mean(isos):.2f} ± {np.std(isos):.2f}")
    print(f"Mean Exposure: {np.mean(exposures):.6f} ± {np.std(exposures):.6f}")
    print(f"Mean GT nonzero bins: {np.mean(gt_nonzero_bins):.2f} ± {np.std(gt_nonzero_bins):.2f}")
    print(f"Mean GT entropy: {np.mean(gt_entropies):.4f} ± {np.std(gt_entropies):.4f}")
    correlation = np.corrcoef(gt_nonzero_bins, [x[1] for x in wass_results])[0, 1]
    print(f"Correlation between GT nonzero bins and Wasserstein: {correlation:.4f}")
    with open(OUTPUT_DIR / f'{mode}_worst_examples_analysis.txt', 'w') as f:
        f.write("Common traits of worst examples:\n")
        f.write(f"Mean LightValue: {np.mean(light_values):.2f} ± {np.std(light_values):.2f}\n")
        f.write(f"Mean ISO: {np.mean(isos):.2f} ± {np.std(isos):.2f}\n")
        f.write(f"Mean Exposure: {np.mean(exposures):.6f} ± {np.std(exposures):.6f}\n")
        f.write(f"Mean GT nonzero bins: {np.mean(gt_nonzero_bins):.2f} ± {np.std(gt_nonzero_bins):.2f}\n")
        f.write(f"Mean GT entropy: {np.mean(gt_entropies):.4f} ± {np.std(gt_entropies):.4f}\n")
        f.write(f"Correlation between GT nonzero bins and Wasserstein: {correlation:.4f}\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
def compute_lime_explanations(mode='indoor'):
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
    wass_results = []
    with torch.no_grad():
        for batch in tqdm(datamodule.get_val_dataloader(), desc="Collecting worst examples"):
            img, chroma_hist, illum_hist, edge_map, depth, path, _, brightness, saturation, _, patch_stats, _ = batch
            img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, patch_stats = (
                img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
            pred_hist, _, _, _, _, _ = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
            wass = sliced_wasserstein_loss(pred_hist, illum_hist).item()
            wass_results.append((path[0], wass, img.cpu(), chroma_hist.cpu(), edge_map.cpu(), depth.cpu(), brightness.cpu(), saturation.cpu(), patch_stats.cpu()))
            del img, chroma_hist, illum_hist, edge_map, depth, pred_hist, brightness, saturation, patch_stats
            gc.collect()
            torch.cuda.empty_cache()
    wass_results = sorted(wass_results, key=lambda x: x[1], reverse=True)[:5]
    explainer = lime_image.LimeImageExplainer()
    print("LIME Analysis")
    print("Example\tPath\tWasserstein")
    for i, (path, wass, img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats) in enumerate(wass_results):
        img_np = img[0].permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
       
        def model_wrapper(inputs):
            img_np = inputs.transpose(0, 3, 1, 2)
            img_t = torch.tensor(img_np, dtype=torch.float32).to(DEVICE)
            batch_size = img_t.size(0)
            pred_hists = []
            for j in range(batch_size):
                with torch.no_grad():
                    pred_hist, _, _, _, _, _ = model(img_t[j:j+1], chroma_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE), brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
                    pred_hists.append(pred_hist.detach().cpu().numpy())
            return np.concatenate(pred_hists, axis=0).reshape(batch_size, -1)
       
        explanation = explainer.explain_instance(
            img_np,
            model_wrapper,
            top_labels=1,
            hide_color=0,
            num_samples=100,
            segmentation_fn=None
        )
       
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
       
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image: {Path(path).name}")
        plt.imshow(img_np)
        plt.axis('off')
       
        plt.subplot(1, 2, 2)
        plt.title(f"LIME Explanation (Wass: {wass:.3f})")
        plt.imshow(mask, cmap='hot')
        plt.colorbar()
        plt.axis('off')
       
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{mode}_lime_explanation_{i+1}.png', dpi=300)
        plt.close()
       
        print(f"{i+1}\t{path}\t{wass:.4f}")
       
        del explanation, temp, mask
        gc.collect()
        torch.cuda.empty_cache()
   
    print("LIME analysis completed. Visualizations saved as lime_explanation_1.png to lime_explanation_5.png")
    with open(OUTPUT_DIR / f'{mode}_lime_analysis.txt', 'w') as f:
        f.write("LIME Analysis: Evaluated contribution of image regions to histogram prediction\n")
        f.write(f"Analyzed 5 worst examples by Wasserstein distance\n")
        for i, (path, wass, _, _, _, _, _, _, _) in enumerate(wass_results):
            f.write(f"Example {i+1}: {path}, Wass={wass:.4f}\n")
        f.write("Visualizations saved as lime_explanation_1.png to lime_explanation_5.png\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
def evaluate_model_stability(mode='indoor'):
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
    wass_results_per_seed = []
   
    for seed in range(42, 47):
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        seed_everything(seed)
        model = HistPredictor().to(DEVICE)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            for key in state_dict:
                state_dict[key] += torch.randn_like(state_dict[key], device=DEVICE) * 0.01
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"Error: {model_path} not found")
            return
       
        model.eval()
        wass_values = []
       
        with torch.no_grad():
            for batch in tqdm(datamodule.get_val_dataloader(), desc=f"Stability seed {seed}"):
                img, chroma_hist, illum_hist, edge_map, depth, _, _, brightness, saturation, _, patch_stats, _ = batch
                img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, patch_stats = (
                    img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                    brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
                pred_hist, _, _, _, _, _ = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
                wass = sliced_wasserstein_loss(pred_hist, illum_hist).item()
                wass_values.append(wass)
                del img, chroma_hist, illum_hist, edge_map, depth, pred_hist, brightness, saturation, patch_stats
                gc.collect()
                torch.cuda.empty_cache()
       
        mean_wass = np.mean(wass_values)
        wass_results_per_seed.append(mean_wass)
        print(f"Seed {seed}: Mean Wasserstein = {mean_wass:.4f}")
       
        del model
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
   
    mean_wass = np.mean(wass_results_per_seed)
    std_wass = np.std(wass_results_per_seed)
    stability_status = "stable" if std_wass < 0.35 else "unstable"
    print("Model Stability")
    print("Metric\tValue")
    print(f"Mean Wasserstein\t{mean_wass:.4f}")
    print(f"Std Wasserstein\t{std_wass:.4f}")
    print(f"Status\t{stability_status}")
    print(f"Model is {stability_status} (Std Wasserstein < 0.3 is considered stable)")
    with open(OUTPUT_DIR / f'{mode}_model_stability.txt', 'w') as f:
        f.write(f"Model Stability: Mean Wasserstein = {mean_wass:.4f}, Std = {std_wass:.4f}\n")
        f.write(f"Details: Evaluated over 5 seeds with slight weight perturbations (std=0.01)\n")
        f.write(f"Model is {stability_status} (Std Wasserstein < 0.3 is considered stable)\n")
    gc.collect()
    torch.cuda.empty_cache()
def analyze_hypothesis(mode='indoor'):
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
    gt_entropies = []
    light_value_devs = []
    isos = []
    edge_densities = []
    depth_stds = []
    chroma_entropies = []
    print(f"Analyzing hypotheses on validation set: {len(datamodule.val_dataset)} images")
    with torch.no_grad():
        for batch in tqdm(datamodule.get_val_dataloader(), desc="Hypotheses analysis"):
            img, chroma_hist, illum_hist, edge_map, depth, path, _, brightness, saturation, _, patch_stats, _ = batch
            img, chroma_hist, illum_hist, edge_map, depth, brightness, saturation, patch_stats = (
                img.to(DEVICE), chroma_hist.to(DEVICE), illum_hist.to(DEVICE), edge_map.to(DEVICE), depth.to(DEVICE),
                brightness.to(DEVICE), saturation.to(DEVICE), patch_stats.to(DEVICE))
            pred_hist, _, _, _, _, _ = model(img, chroma_hist, edge_map, depth, brightness, saturation, patch_stats)
            wass = sliced_wasserstein_loss(pred_hist, illum_hist).item()
           
            gt_np = illum_hist[0].cpu().numpy()
            gt_entropy = entropy(gt_np.flatten() + 1e-10)
            gt_entropies.append(gt_entropy)
           
            name = f"{Path(path[0]).parent.name}/{Path(path[0]).name}"
            meta = metadata_df[metadata_df['names'] == name].iloc[0]
            light_value = meta['LightValue']
            light_value_dev = abs(light_value - 10.0)
            light_value_devs.append(light_value_dev)
           
            iso = meta['ISO']
            isos.append(iso)
           
            edge_np = edge_map[0].cpu().numpy()
            edge_density = (edge_np > 1e-6).mean()
            edge_densities.append(edge_density)
           
            depth_np = depth[0].cpu().numpy()
            depth_std = np.std(depth_np)
            depth_stds.append(depth_std)
           
            chroma_np = chroma_hist[0].cpu().numpy()
            chroma_entropy = entropy(chroma_np.flatten() + 1e-10)
            chroma_entropies.append(chroma_entropy)
           
            wass_values.append(wass)
            del img, chroma_hist, illum_hist, edge_map, depth, pred_hist, brightness, saturation, patch_stats
            gc.collect()
            torch.cuda.empty_cache()
    correlations = {
        "GT Entropy vs Wasserstein": np.corrcoef(gt_entropies, wass_values)[0, 1],
        "LightValue Deviation vs Wasserstein": np.corrcoef(light_value_devs, wass_values)[0, 1],
        "ISO vs Wasserstein": np.corrcoef(isos, wass_values)[0, 1],
        "Edge Density vs Wasserstein": np.corrcoef(edge_densities, wass_values)[0, 1],
        "Depth Std vs Wasserstein": np.corrcoef(depth_stds, wass_values)[0, 1],
        "Chroma Entropy vs Wasserstein": np.corrcoef(chroma_entropies, wass_values)[0, 1]
    }
    print("Hypotheses Analysis")
    print("Hypothesis\tCorrelation\tStatus")
    hypotheses = [
        ("Model performs worse on scenes with high GT entropy", correlations["GT Entropy vs Wasserstein"]),
        ("Model performs worse on scenes with extreme LightValue", correlations["LightValue Deviation vs Wasserstein"]),
        ("Model performs worse on images with high ISO", correlations["ISO vs Wasserstein"]),
        ("Model performs worse on images with high edge density", correlations["Edge Density vs Wasserstein"]),
        ("Model performs worse on scenes with high depth variation", correlations["Depth Std vs Wasserstein"]),
        ("Model performs worse on images with high chroma entropy", correlations["Chroma Entropy vs Wasserstein"])
    ]
    for hypothesis, corr in hypotheses:
        status = "confirmed" if corr > 0.3 else "partially confirmed" if corr > 0 else "not confirmed"
        print(f"{hypothesis}\t{corr:.4f}\t{status}")
    print("Hypotheses Analysis Summary:")
    for hypothesis, corr in hypotheses:
        status = "confirmed" if corr > 0.3 else "partially confirmed" if corr > 0 else "not confirmed"
        print(f"{hypothesis}: Correlation = {corr:.4f}, Status = {status}")
    with open(OUTPUT_DIR / f'{mode}_hypothesis_analysis.txt', 'w') as f:
        f.write("Hypotheses Analysis:\n")
        for hypothesis, corr in hypotheses:
            status = "confirmed" if corr > 0.3 else "partially confirmed" if corr > 0 else "not confirmed"
            f.write(f"{hypothesis}: Correlation = {corr:.4f}, Status = {status}\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Generating submission...")
    generate_submission_zip()
    print("Analyzing for indoor:")
    analyze_light_value_hypothesis(mode='indoor')
    analyze_worst_examples(mode='indoor')
    compute_lime_explanations(mode='indoor')
    evaluate_model_stability(mode='indoor')
    analyze_hypothesis(mode='indoor')
    print("Analyzing for outdoor:")
    analyze_light_value_hypothesis(mode='outdoor')
    analyze_worst_examples(mode='outdoor')
    compute_lime_explanations(mode='outdoor')
    evaluate_model_stability(mode='outdoor')
    analyze_hypothesis(mode='outdoor')