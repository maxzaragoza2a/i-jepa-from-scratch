"""Teste le modèle sur des images hors CIFAR-10 déposées dans ce dossier.

Chaque image (.jpg, .jpeg, .png, .bmp, .webp) est recadrée au carré (centre),
réduite en 32x32 comme CIFAR-10, puis traitée comme dans
result_analysis/predicted_fill.py : un masque, la prédiction des blocs cibles,
et le retrieval des patchs les plus proches parmi les images de test CIFAR-10.

Le score « vrai patch en top-1 » est aussi calculé : les patchs de l'image
testée sont ajoutés à la banque le temps de cette mesure.

Usage (depuis la racine du projet) :
    python experimentation/test_image.py
    python experimentation/test_image.py --seed 3
    python experimentation/test_image.py chat.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "result_analysis"))

from jepa.config import JEPAConfig  # noqa: E402
from jepa.masking import sample_masks  # noqa: E402
from predicted_fill import (PatchBank, fill_images, load_model,  # noqa: E402
                            mask_summary, plot_fills, predicted_tokens)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path, config):
    """Fichier -> (32, 32, 3) dans [0, 1], recadré au centre puis réduit."""
    img = Image.open(path).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((config.image_size, config.image_size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def standardise(images, config):
    return ((images - np.array(config.mean)) / np.array(config.std)).astype(np.float32)


def self_hits(model, bank, images, context_indices, target_indices):
    """Combien de vrais patchs sont le plus proche voisin de leur prédiction,
    dans la banque CIFAR complétée des patchs de l'image elle-même."""
    config = bank.config
    own = np.asarray(model.target_encoder(images, training=False))  # (B, 64, D)
    preds = predicted_tokens(model, images, context_indices, target_indices)
    hits = total = 0
    for i in range(len(images)):
        own_sq = (own[i] ** 2).sum(axis=1)
        for token, pred in preds.items():
            best_bank = bank.distances(pred[i]).min()
            own_dist = own_sq - 2 * own[i] @ pred[i]
            hits += int(np.argmin(own_dist) == token and own_dist[token] < best_bank)
            total += 1
    return hits, total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="images à tester (défaut : tout le dossier)")
    parser.add_argument("--seed", type=int, default=0, help="tirage du masque")
    parser.add_argument("--out", default=str(HERE / "predictions.png"))
    args = parser.parse_args()

    paths = ([HERE / f for f in args.files] if args.files else
             sorted(p for p in HERE.rglob("*") if p.suffix.lower() in EXTENSIONS
                    and p.resolve() != Path(args.out).resolve()))
    if not paths:
        sys.exit(f"Aucune image trouvée dans {HERE}. Déposez-y un .jpg ou un .png.")

    config = JEPAConfig()
    rng = np.random.default_rng(args.seed)
    originals = np.stack([load_image(p, config) for p in paths])
    images = standardise(originals, config)

    model = load_model(config, images, rng)
    bank = PatchBank(model, config)
    context_indices, target_indices = sample_masks(config, rng)

    fills, _, _ = fill_images(model, bank, images, context_indices, target_indices)
    hits, total = self_hits(model, bank, images, context_indices, target_indices)

    title = (f"I-JEPA, images hors CIFAR-10  --  seed {args.seed}, "
             f"{mask_summary(context_indices, target_indices, config)}\n"
             f"chaque patch masqué = patch CIFAR-10 de test dont l'embedding est le plus proche "
             f"de la prédiction\nvrai patch retrouvé en 1er (parmi {len(bank) // 1000}k + les "
             f"siens) : {100 * hits / total:.0f} % ({hits}/{total})")
    plot_fills(originals, fills, context_indices, target_indices, config, title, args.out,
               names=[p.name for p in paths])
    print(f"vrai patch en top-1 : {hits}/{total} ({100 * hits / total:.1f} %)")


if __name__ == "__main__":
    main()
