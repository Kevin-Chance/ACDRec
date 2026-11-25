def fuse_scores(text_items, text_scores, image_items, image_scores):
    if len(text_items) != len(image_items) or len(text_items) != len(text_scores) or len(image_items) != len(image_scores):
        raise ValueError('Both branches must have the same number of item scores')
    item_id2score = {}
    for i in range(len(text_items)):
        for item_id, score in [(text_items[i], text_scores[i]), (image_items[i], image_scores[i])]:
            if item_id == -1:
                score = -1000
            if item_id in item_id2score and item_id != -1:
                item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
            else:
                item_id2score[item_id] = score
    return sorted(item_id2score.items(), key=lambda pair: pair[1], reverse=True)
