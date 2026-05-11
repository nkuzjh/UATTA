import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.search_dataset import search_test_dataset
from torch.utils.data import Dataset

def create_test_dataset(config):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    test_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = search_test_dataset(config, test_transform)

    return test_dataset

def create_test_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
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
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    return loaders

class search_tta_dataset(Dataset):
    def __init__(self, config, tta_transform, text_to_image_similarity, image_embeds, text_embeds, text_atts, recall_types, selected_query_indices, uncertainties, top1_probabilities, reciprocal_probabilities):

        self.config = config
        self.text_to_image_similarity = text_to_image_similarity
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds
        self.text_atts = text_atts
        self.uncertainties = uncertainties
        self.top1_probabilities = top1_probabilities
        self.reciprocal_probabilities = reciprocal_probabilities

        if len(selected_query_indices) != len(self.text_to_image_similarity):
            self.text_to_image_similarity = text_to_image_similarity[selected_query_indices]
            self.text_embeds = text_embeds[selected_query_indices]
            self.text_atts = text_atts[selected_query_indices]
            self.uncertainties = [uncertainties[i] for i in selected_query_indices]
            self.top1_probabilities = [top1_probabilities[i] for i in selected_query_indices]
            self.reciprocal_probabilities = [reciprocal_probabilities[i] for i in selected_query_indices]

    def __len__(self):
        return len(self.text_to_image_similarity)

    def __getitem__(self, index):

        _, topk_image_indices = self.text_to_image_similarity[index].topk(k=self.config['k_tta'], dim=0)
        encoder_output = self.image_embeds[topk_image_indices]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long)
        text_embeds = self.text_embeds[index].repeat(self.config['k_tta'], 1, 1)
        text_atts = self.text_atts[index].repeat(self.config['k_tta'], 1)
        uncertainty = self.uncertainties[index]
        top1_probability = self.top1_probabilities[index]
        reciprocal_probability = self.reciprocal_probabilities[index]

        return encoder_output, encoder_att, text_embeds, text_atts, uncertainty, top1_probability, reciprocal_probability

def create_tta_dataset(config, text_to_image_similarity, image_embeds, text_embeds, text_atts, recall_types, selected_query_indices, uncertainties, top1_probabilities, reciprocal_probabilities):

    tta_transform = None

    if config.get('is_image_augmentation', False):
        raise NotImplementedError("Image augmentation is not part of the released PAB SOTA reproduction path.")
    else:
        tta_dataset = search_tta_dataset(config, tta_transform, text_to_image_similarity, image_embeds, text_embeds, text_atts, recall_types, selected_query_indices, uncertainties, top1_probabilities, reciprocal_probabilities)

    return tta_dataset

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
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    return loaders
