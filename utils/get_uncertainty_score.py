from utils.entropy import get_entropy_scores
from utils.class_entropy import get_class_entropy_scores
from utils.distance_to_half_score import get_distance_to_half_scores
from utils.margin import get_margin_scores
from utils.class_score import get_class_scores
from utils.class_margin import get_class_margin
from config import get_config

Config = get_config()



def get_uncertainty_scores(model, unlabel_dataset, label_dataset, device):

    if Config.Uncertainty_Score == 'Entropy':
        return get_entropy_scores(
            model=model,
            dataset=unlabel_dataset,
            label_dataset = label_dataset,
            device=device,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
    elif Config.Uncertainty_Score == 'Class_Entropy':
        return get_class_entropy_scores(
            model=model,
            dataset=unlabel_dataset,
            label_dataset = label_dataset,
            device=device,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
    elif Config.Uncertainty_Score == 'Distance_to_half':
        return get_distance_to_half_scores(
            model=model,
            dataset=unlabel_dataset,
            label_dataset = label_dataset,
            device=device,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
    elif Config.Uncertainty_Score == 'Margin':
        return get_margin_scores(
            model=model,
            dataset=unlabel_dataset,
            label_dataset = label_dataset,
            device=device,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
    elif Config.Uncertainty_Score == 'class_score':
        return get_class_scores(
            model=model,
            unlabel_dataset = unlabel_dataset,
            label_dataset = label_dataset,
            device=device,
            num_classes = 47,
            
            
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
    elif Config.Uncertainty_Score == 'class_margin':
        return get_class_margin(
            model=model,
            unlabel_dataset=unlabel_dataset,
            label_dataset = label_dataset,
            device=device,
            num_classes = 10,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
    else:
        raise ValueError(f"Unsupported Uncertainty_Score: {Config.Uncertainty_Score}")