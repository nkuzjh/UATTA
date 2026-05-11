from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class IRRA_tta_dataset(Dataset):
    def __init__(
        self,
        config, test_transforms,
        topk_image_indices, query_ids, gallery_ids, captions, images,
        recall_types,
        selected_query_indices,
        uncertainties,
        top1_probabilities,
        reciprocal_probabilities
    ):
        self.config = config
        self.transform = test_transforms
        self.topk_image_indices = topk_image_indices
        self.query_ids = query_ids
        self.gallery_ids = gallery_ids
        self.captions = captions
        self.images = images
        self.uncertainties = uncertainties
        self.top1_probabilities = top1_probabilities
        self.reciprocal_probabilities = reciprocal_probabilities

        if len(selected_query_indices) != len(self.topk_image_indices):
            self.topk_image_indices = topk_image_indices[selected_query_indices]
            self.query_ids = query_ids[selected_query_indices]
            self.captions = captions[selected_query_indices]
            self.uncertainties = [uncertainties[i] for i in selected_query_indices]
            self.top1_probabilities = [top1_probabilities[i] for i in selected_query_indices]
            self.reciprocal_probabilities = [reciprocal_probabilities[i] for i in selected_query_indices]

    def __len__(self):
        return len(self.topk_image_indices)

    def __getitem__(self, index):
        topk_indices = self.topk_image_indices[index]
        gallery_ids_topk = self.gallery_ids[topk_indices]
        images_topk = self.images[topk_indices]

        query_id = self.query_ids[index]
        caption = self.captions[index]
        uncertainty = self.uncertainties[index]
        top1_probability = self.top1_probabilities[index]
        reciprocal_probability = self.reciprocal_probabilities[index]

        return gallery_ids_topk, images_topk, query_id, caption, uncertainty, top1_probability, reciprocal_probability

def create_tta_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, bs, n_worker, is_train, collate_fn in zip(datasets, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    return loaders
