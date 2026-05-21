15. 验证集评估函数

COCO 每张图有多个 caption。你需要按 image_id 聚合 reference captions。

def build_image_references(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    refs = {}
    for ann in data["annotations"]:
        file_name = image_id_to_file[ann["image_id"]]
        refs.setdefault(file_name, []).append(ann["caption"])

    return refs

构造一个只按 image 遍历的 validation dataset：

class CocoImageDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.images = [img["file_name"] for img in data["images"]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        path = os.path.join(self.image_dir, file_name)

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return file_name, image

评估：

@torch.no_grad()
def evaluate_model(model, image_loader, references_dict, vocab, device, max_len=30):
    model.eval()

    hypotheses = []
    references = []

    for file_names, images in tqdm(image_loader):
        images = images.to(device)

        for i in range(images.size(0)):
            image = images[i]
            file_name = file_names[i]

            hyp = generate_caption_system(
                model=model,
                image=image,
                vocab=vocab,
                device=device,
                max_len=max_len
            )

            hypotheses.append(hyp)
            references.append(references_dict[file_name])

    return compute_metrics(references, hypotheses)
16. 你的实验表应该这样设计

你最后应该得到一个类似原论文 Table 2 的表：

Model	Encoder	Decoder	MemTorch	BLEU1	BLEU2	BLEU3	BLEU4	ROUGE-L	METEOR
Ref-T	ResNet50	2-layer Transformer	No	...	...	...	...	...	...
CMM-T it-10	ResNet50	2-layer Transformer	Yes	...	...	...	...	...	...
CMM-T it-9	ResNet50	2-layer Transformer	Yes	...	...	...	...	...	...
CMM-T it-8	ResNet50	2-layer Transformer	Yes	...	...	...	...	...	...
CMM-T it-7	ResNet50	2-layer Transformer	Yes	...	...	...	...	...	...
CMM-T it-6	ResNet50	2-layer Transformer	Yes	...	...	...	...	...	...

这样你就能直接和原论文的实验逻辑对齐：
软件参考模型 vs 忆阻器写入误差模型 vs 不同硬件条件。