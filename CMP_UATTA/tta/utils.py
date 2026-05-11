
import torch
import torch.nn.functional as F

def sample_selection_itc(text_to_image_similarity, image_to_text_similarity):
    selected_query_indices = []
    for query_index, text_to_image_scores in enumerate(text_to_image_similarity):
        _, top1_image_index = text_to_image_scores.topk(k=1, dim=0)
        _, top1_query_index = image_to_text_similarity[top1_image_index][0].topk(k=1, dim=0)
        if top1_query_index == query_index:
            selected_query_indices.append(query_index)

    return selected_query_indices

def sample_selection_topk_itc(text_to_image_similarity, image_to_text_similarity, k_sample_selection):
    selected_query_indices = []
    for query_index, text_to_image_scores in enumerate(text_to_image_similarity):
        _, topk_image_indices = text_to_image_scores.topk(k=k_sample_selection, dim=-1)
        _, topk_query_indices = image_to_text_similarity[topk_image_indices].topk(k=k_sample_selection, dim=-1)
        if query_index in topk_query_indices.reshape(-1).tolist():
            selected_query_indices.append(query_index)

    return selected_query_indices

def compute_uncertainty_itc(config, text_to_image_similarity, image_to_text_similarity):
    k_test = config['k_test']
    uncertainty_temper = config.get('uncertainty_temper', 1.0)
    uncertainty_t2i_temper = config.get('uncertainty_t2i_temper', 1.0)
    uncertainty_i2t_temper = config.get('uncertainty_i2t_temper', 1.0)

    inverse_recall_uncertainties = []
    diff_ratio_uncertainties = []
    log_diff_uncertainties = []
    top1_probabilities = []
    reciprocal_probabilities = []
    for query_index, text_to_image_scores in enumerate(text_to_image_similarity):

        topk_text_to_image_scores, topk_image_indices = text_to_image_scores.topk(k=k_test, dim=0)
        top1_probability = F.softmax(topk_text_to_image_scores * uncertainty_t2i_temper, dim=0)[0]

        reciprocal_probability = torch.zeros([])
        topk_image_to_text_scores, topk_query_indices = image_to_text_similarity[topk_image_indices[0]].topk(k=k_test, dim=0)
        if query_index in topk_query_indices:
            query_rank = torch.where(topk_query_indices == query_index)[0][0]
            reciprocal_probability = F.softmax(topk_image_to_text_scores * uncertainty_i2t_temper, dim=0)[query_rank]

        uncertainty1 = torch.exp((1 - (top1_probability + reciprocal_probability) / 2) * uncertainty_temper)
        uncertainty2 = torch.exp(torch.abs(top1_probability - reciprocal_probability) / ((top1_probability + reciprocal_probability) / 2) * uncertainty_temper)
        uncertainty3 = torch.abs(torch.log(top1_probability + 1e-2) - torch.log(reciprocal_probability + 1e-2))

        inverse_recall_uncertainties.append(uncertainty1)
        diff_ratio_uncertainties.append(uncertainty2)
        log_diff_uncertainties.append(uncertainty3)
        top1_probabilities.append(top1_probability)
        reciprocal_probabilities.append(reciprocal_probability)
    return inverse_recall_uncertainties, diff_ratio_uncertainties, log_diff_uncertainties, top1_probabilities, reciprocal_probabilities

def preprocess_tta_coefficients(config, text_to_image_similarity):
    print(f"     preprocess_tta_coefficients  start")

    recall_types = []

    print(f"     sample selection ...")
    if config.get('sample_selection', 'all') == 'top1':
        selected_query_indices = sample_selection_itc(text_to_image_similarity, text_to_image_similarity.t())
    elif config.get('sample_selection', 'all') == 'topk':
        k_sample_selection = config.get('k_sample_selection', 5)
        selected_query_indices = sample_selection_topk_itc(text_to_image_similarity, text_to_image_similarity.t(), k_sample_selection)
    else:
        selected_query_indices = torch.arange(0, text_to_image_similarity.size(0))
    print("     number of sample after sample_selection: {}".format(len(selected_query_indices)))

    print(f"     uncertainty ...")
    if config.get('uncertainty', None) is not None:
        uncertainty_similarity = text_to_image_similarity / config.get('temperature', 1.0)
        inverse_recall_uncertainties, diff_ratio_uncertainties, log_diff_uncertainties, top1_probabilities, reciprocal_probabilities = compute_uncertainty_itc(config, uncertainty_similarity, uncertainty_similarity.t())

    if config.get('uncertainty', None) == 'inversed_recall_proba':
        uncertainties = inverse_recall_uncertainties
    elif config.get('uncertainty', None) == 'diff_div_mean':
        uncertainties = diff_ratio_uncertainties
    elif config.get('uncertainty', None) == 'abs_diff_log':
        uncertainties = log_diff_uncertainties
    else:
        uncertainties = torch.ones(text_to_image_similarity.size(0))
        top1_probabilities = torch.ones(text_to_image_similarity.size(0))
        reciprocal_probabilities = torch.ones(text_to_image_similarity.size(0))

    print(f"     preprocess_tta_coefficients  end")
    return recall_types, selected_query_indices, uncertainties, top1_probabilities, reciprocal_probabilities
