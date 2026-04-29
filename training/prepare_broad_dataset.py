import os
import yaml
from pathlib import Path
from shutil import copy2

BROAD_LABEL_MAP = {
    'Aerosols': 'Other',
    'Aluminum can': 'Metal',
    'Aluminum caps': 'Metal',
    'Cardboard': 'Paper',
    'Cellulose': 'Paper',
    'Ceramic': 'Other',
    'Combined plastic': 'Plastic',
    'Container for household chemicals': 'Other',
    'Disposable tableware': 'Other',
    'Electronics': 'Other',
    'Foil': 'Metal',
    'Furniture': 'Other',
    'Glass bottle': 'Glass',
    'Iron utensils': 'Metal',
    'Liquid': 'Other',
    'Metal shavings': 'Metal',
    'Milk bottle': 'Plastic',
    'Organic': 'Organic',
    'Paper': 'Paper',
    'Paper bag': 'Paper',
    'Paper cups': 'Paper',
    'Paper shavings': 'Paper',
    'Papier mache': 'Paper',
    'Plastic bag': 'Plastic',
    'Plastic bottle': 'Plastic',
    'Plastic can': 'Plastic',
    'Plastic canister': 'Plastic',
    'Plastic caps': 'Plastic',
    'Plastic cup': 'Plastic',
    'Plastic shaker': 'Plastic',
    'Plastic shavings': 'Plastic',
    'Plastic toys': 'Plastic',
    'Postal packaging': 'Other',
    'Printing industry': 'Other',
    'Scrap metal': 'Metal',
    'Stretch film': 'Plastic',
    'Tetra pack': 'Other',
    'Textile': 'Other',
    'Tin': 'Metal',
    'Unknown plastic': 'Plastic',
    'Wood': 'Other',
    'Zip plastic bag': 'Plastic',
}

BROAD_CLASSES = ['Plastic', 'Metal', 'Paper', 'Glass', 'Organic', 'Other']


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def hardlink_file(src, dst):
    try:
        if os.path.exists(dst):
            return
        os.link(src, dst)
    except OSError:
        copy2(src, dst)


def remap_label_file(src_path, dst_path, label_names, name_map):
    with open(src_path, 'r', encoding='utf-8', errors='ignore') as src:
        lines = [line.strip() for line in src if line.strip()]

    mapped_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls >= len(label_names):
            continue
        original_name = label_names[cls]
        mapped_name = name_map.get(original_name)
        if mapped_name is None:
            continue
        mapped_cls = BROAD_CLASSES.index(mapped_name)
        mapped_lines.append(' '.join([str(mapped_cls)] + parts[1:]))

    if mapped_lines:
        with open(dst_path, 'w', encoding='utf-8') as dst:
            dst.write('\n'.join(mapped_lines) + '\n')


def prepare_broad_dataset(source_yaml='datasets/data.yaml', output_root='datasets/broad'):
    source_yaml = Path(source_yaml)
    output_root = Path(output_root)
    config = load_yaml(source_yaml)

    if 'names' not in config or 'train' not in config or 'val' not in config:
        raise ValueError(f'Invalid source dataset config: {source_yaml}')

    label_names = config['names']
    ensure_dir(output_root)

    for split in ['train', 'valid', 'test']:
        src_images = (source_yaml.parent / split / 'images').resolve()
        src_labels = (source_yaml.parent / split / 'labels').resolve()
        dst_images = (output_root / split / 'images').resolve()
        dst_labels = (output_root / split / 'labels').resolve()
        ensure_dir(dst_images)
        ensure_dir(dst_labels)

        if src_images.is_dir():
            for img_path in src_images.glob('*'):
                if img_path.is_file():
                    hardlink_file(str(img_path), str(dst_images / img_path.name))

        if src_labels.is_dir():
            for label_path in src_labels.glob('*.txt'):
                remap_label_file(label_path, dst_labels / label_path.name, label_names, BROAD_LABEL_MAP)

    broad_config = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(BROAD_CLASSES),
        'names': BROAD_CLASSES,
    }
    save_yaml(output_root / 'data.yaml', broad_config)
    print(f'Created broad dataset at {output_root}')


if __name__ == '__main__':
    prepare_broad_dataset()
