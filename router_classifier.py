"""
router_classifier.py  ─  TRACE Router Classifier
═══════════════════════════════════════════════════════════════════════════════
A lightweight multi-label classifier trained on your own tool-call history.
It predicts WHICH tool(s) to call for a given user message so the heavy LLM
(SmolLM3 / Qwen / etc.) only needs to fill in arguments — not pick the tool.

Architecture
────────────
  Input message
       │
  Embedding  (sentence-transformers all-MiniLM-L6-v2, ~22 MB)
       │
  RouterHead (2-layer MLP)
       │
  Sigmoid → per-tool probability
       │
  Threshold → predicted tool set   ─→  skip LLM tool selection, just fill args
                                        OR pass predicted tools to LLM as hint

Data format (JSONL)
────────────
Each line in your training file must look like:

  {"message": "open chrome and go to github.com",  "tools": ["run_command", "web_fetch"]}
  {"message": "what time is it?",                   "tools": ["run_command"]}
  {"message": "summarise this pdf",                 "tools": ["read_file", "run_command"]}
  {"message": "tell me a joke",                     "tools": []}          ← no tool needed

Colab usage
────────────
  !pip install -q sentence-transformers scikit-learn tqdm

  from router_classifier import RouterClassifier

  rc = RouterClassifier()
  rc.train("tool_calls.jsonl")          # first time
  rc.save("router.pt")                  # persist

  # later / inference
  rc = RouterClassifier.load("router.pt")
  print(rc.predict("open the browser"))
  # → ['run_command', 'web_fetch']

Integration with v1_colab_brain.py
────────────
  # At the top of brain_loop_local(), after receiving qwen_tools:
  from router_classifier import RouterClassifier
  rc = RouterClassifier.load("router.pt")   # if pre-trained

  # Inside run_turn_local(), before calling _generate():
  predicted_tools = rc.predict(message)
  # Filter qwen_tools to predicted set (speeds up + reduces hallucination)
  filtered = [t for t in qwen_tools if t["function"]["name"] in predicted_tools] or qwen_tools
  output = _generate(messages, filtered)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import re
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("router_classifier")

# ─── Install guard ───────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Run:  !pip install -q sentence-transformers\n"
        "Then restart the runtime and re-import."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file, tolerating blank / comment lines."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping bad JSON at line %d: %s", lineno, e)
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def build_label_vocab(records: list[dict]) -> list[str]:
    """Return sorted list of all unique tool names found in records."""
    all_tools: set[str] = set()
    for rec in records:
        for t in rec.get("tools", []):
            all_tools.add(t)
    vocab = sorted(all_tools)
    logger.info("Label vocabulary (%d tools): %s", len(vocab), vocab)
    return vocab


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PyTorch dataset
# ═══════════════════════════════════════════════════════════════════════════════

class ToolDataset(Dataset):
    """Pre-computed embeddings + multi-hot labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.Y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Router head (MLP)
# ═══════════════════════════════════════════════════════════════════════════════

class RouterHead(nn.Module):
    """
    2-layer MLP with BatchNorm + Dropout.
    Input  : sentence embedding  (384-dim for MiniLM)
    Output : per-tool logit      (num_tools-dim)
    """

    def __init__(self, embed_dim: int, num_tools: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_tools),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main class
# ═══════════════════════════════════════════════════════════════════════════════

class RouterClassifier:
    """
    Lightweight tool-call router classifier.

    Parameters
    ----------
    encoder_model : HuggingFace sentence-transformer model name.
                    'all-MiniLM-L6-v2' is fast (~22 MB) and good.
    threshold     : Sigmoid probability cutoff for positive prediction.
                    Lower → more tools recalled, Higher → more precise.
    hidden        : Hidden units in the MLP head.
    device        : 'cuda' / 'cpu' (auto-detected if None).
    """

    ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        encoder_model: str = ENCODER_MODEL,
        threshold: float = 0.45,
        hidden: int = 256,
        device: Optional[str] = None,
    ):
        self.encoder_model_name = encoder_model
        self.threshold = threshold
        self.hidden = hidden
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("RouterClassifier device: %s", self.device)

        self._encoder: Optional[SentenceTransformer] = None
        self._head: Optional[RouterHead] = None
        self.label_vocab: list[str] = []
        self.embed_dim: int = 0

    # ── lazy encoder load ─────────────────────────────────────────────────────
    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info("Loading sentence encoder: %s", self.encoder_model_name)
            self._encoder = SentenceTransformer(self.encoder_model_name, device=self.device)
        return self._encoder

    # ── encode ────────────────────────────────────────────────────────────────
    def _encode(self, texts: list[str], batch_size: int = 128, show_progress: bool = True) -> np.ndarray:
        return self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # ── multi-hot encode labels ────────────────────────────────────────────────
    def _to_multihot(self, tool_lists: list[list[str]]) -> np.ndarray:
        idx = {t: i for i, t in enumerate(self.label_vocab)}
        Y = np.zeros((len(tool_lists), len(self.label_vocab)), dtype=np.float32)
        for row, tools in enumerate(tool_lists):
            for t in tools:
                if t in idx:
                    Y[row, idx[t]] = 1.0
                else:
                    logger.warning("Tool '%s' not in vocab — skipped", t)
        return Y

    # ── train ─────────────────────────────────────────────────────────────────
    def train(
        self,
        data_path: str | Path,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 3e-4,
        val_split: float = 0.1,
        pos_weight_scale: float = 2.0,
        seed: int = 42,
    ) -> dict:
        """
        Train the router head on JSONL tool-call data.

        Returns a dict with training history (loss, val_f1 per epoch).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Load & split ──────────────────────────────────────────────────────
        records = load_jsonl(data_path)
        if len(records) < 10:
            raise ValueError("Need at least 10 training examples.")

        messages = [r["message"] for r in records]
        tool_lists = [r.get("tools", []) for r in records]
        self.label_vocab = build_label_vocab(records)

        if not self.label_vocab:
            raise ValueError("No tools found in data. Check your JSONL format.")

        # ── Encode all messages ───────────────────────────────────────────────
        logger.info("Encoding %d messages …", len(messages))
        X = self._encode(messages, batch_size=128)
        Y = self._to_multihot(tool_lists)
        self.embed_dim = X.shape[1]

        # ── Train/val split ───────────────────────────────────────────────────
        n_val = max(1, int(len(X) * val_split))
        idx = np.random.permutation(len(X))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        train_ds = ToolDataset(X[train_idx], Y[train_idx])
        val_ds   = ToolDataset(X[val_idx],   Y[val_idx])

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

        # ── Model ─────────────────────────────────────────────────────────────
        self._head = RouterHead(
            embed_dim=self.embed_dim,
            num_tools=len(self.label_vocab),
            hidden=self.hidden,
        ).to(self.device)

        # Compute per-class pos_weight to handle class imbalance
        pos_counts = Y[train_idx].sum(axis=0) + 1
        neg_counts = len(train_idx) - pos_counts + 1
        pw = torch.tensor(neg_counts / pos_counts * pos_weight_scale, dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.AdamW(self._head.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_dl),
            epochs=epochs,
        )

        history = {"train_loss": [], "val_f1": []}
        best_f1 = -1.0
        best_state = None

        logger.info(
            "Training RouterHead | tools=%d | embed_dim=%d | train=%d | val=%d",
            len(self.label_vocab), self.embed_dim, len(train_ds), len(val_ds),
        )

        for epoch in range(1, epochs + 1):
            # ── train step ────────────────────────────────────────────────────
            self._head.train()
            epoch_loss = 0.0
            for Xb, Yb in train_dl:
                Xb, Yb = Xb.to(self.device), Yb.to(self.device)
                optimizer.zero_grad()
                logits = self._head(Xb)
                loss = criterion(logits, Yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * len(Xb)
            epoch_loss /= len(train_ds)

            # ── val step ──────────────────────────────────────────────────────
            f1 = self._eval_f1(val_dl)
            history["train_loss"].append(round(epoch_loss, 4))
            history["val_f1"].append(round(f1, 4))

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in self._head.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  loss=%.4f  val_f1=%.4f  (best=%.4f)",
                    epoch, epochs, epoch_loss, f1, best_f1,
                )

        # Restore best weights
        if best_state:
            self._head.load_state_dict(best_state)
        logger.info("Training complete. Best val F1: %.4f", best_f1)
        return history

    # ── eval F1 ───────────────────────────────────────────────────────────────
    def _eval_f1(self, dl: DataLoader) -> float:
        """Compute micro-averaged F1 on a dataloader."""
        self._head.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for Xb, Yb in dl:
                Xb = Xb.to(self.device)
                logits = self._head(Xb)
                preds = (torch.sigmoid(logits) >= self.threshold).cpu().numpy().astype(int)
                truth = Yb.numpy().astype(int)
                tp += (preds & truth).sum()
                fp += (preds & ~truth).sum()
                fn += (~preds & truth).sum()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        return 2 * precision * recall / (precision + recall + 1e-9)

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, message: str | list[str], threshold: Optional[float] = None) -> list[str] | list[list[str]]:
        """
        Predict tool(s) for a message (or list of messages).

        Returns
        -------
        Single message  → list of tool names  e.g. ['run_command', 'web_fetch']
        List of messages → list of lists
        """
        if self._head is None or not self.label_vocab:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        single = isinstance(message, str)
        if single:
            message = [message]

        thr = threshold if threshold is not None else self.threshold
        embs = self._encode(message, show_progress=False)

        self._head.eval()
        with torch.no_grad():
            X = torch.tensor(embs, dtype=torch.float32).to(self.device)
            probs = torch.sigmoid(self._head(X)).cpu().numpy()

        results = []
        for row in probs:
            results.append([self.label_vocab[i] for i, p in enumerate(row) if p >= thr])

        return results[0] if single else results

    # ── predict with confidence scores ────────────────────────────────────────
    def predict_scores(self, message: str) -> dict[str, float]:
        """Return {tool_name: probability} for all tools."""
        if self._head is None or not self.label_vocab:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        emb = self._encode([message], show_progress=False)
        self._head.eval()
        with torch.no_grad():
            X = torch.tensor(emb, dtype=torch.float32).to(self.device)
            probs = torch.sigmoid(self._head(X)).cpu().numpy()[0]

        return dict(sorted(zip(self.label_vocab, probs.tolist()), key=lambda x: -x[1]))

    # ── save ──────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Save model weights + vocab + config to a .pt file."""
        path = Path(path)
        payload = {
            "head_state_dict": self._head.state_dict(),
            "label_vocab": self.label_vocab,
            "embed_dim": self.embed_dim,
            "hidden": self.hidden,
            "threshold": self.threshold,
            "encoder_model_name": self.encoder_model_name,
        }
        torch.save(payload, path)
        logger.info("Saved router to %s (%.1f KB)", path, path.stat().st_size / 1024)

    # ── load ──────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "RouterClassifier":
        """Load a saved router classifier."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)

        rc = cls(
            encoder_model=payload["encoder_model_name"],
            threshold=payload["threshold"],
            hidden=payload["hidden"],
            device=device,
        )
        rc.label_vocab = payload["label_vocab"]
        rc.embed_dim   = payload["embed_dim"]
        rc._head = RouterHead(
            embed_dim=rc.embed_dim,
            num_tools=len(rc.label_vocab),
            hidden=rc.hidden,
        ).to(rc.device)
        rc._head.load_state_dict(payload["head_state_dict"])
        rc._head.eval()
        logger.info(
            "Loaded router from %s | tools=%d | threshold=%.2f",
            path, len(rc.label_vocab), rc.threshold,
        )
        return rc

    # ── fine-tune on new data ─────────────────────────────────────────────────
    def fine_tune(
        self,
        new_data_path: str | Path,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 32,
    ) -> None:
        """
        Incrementally fine-tune on new examples without forgetting old vocab.
        New tool names found in new_data_path are appended to label_vocab and
        the head output layer is expanded accordingly.
        """
        if self._head is None:
            raise RuntimeError("Train a base model first before fine-tuning.")

        records = load_jsonl(new_data_path)
        messages = [r["message"] for r in records]
        tool_lists = [r.get("tools", []) for r in records]

        # Expand vocab if new tools appeared
        new_tools = sorted({t for tl in tool_lists for t in tl} - set(self.label_vocab))
        if new_tools:
            logger.info("Adding %d new tools to vocab: %s", len(new_tools), new_tools)
            old_n = len(self.label_vocab)
            self.label_vocab.extend(new_tools)
            new_n = len(self.label_vocab)
            # Expand final linear layer weights
            old_fc = self._head.net[-1]
            new_fc = nn.Linear(old_fc.in_features, new_n).to(self.device)
            with torch.no_grad():
                new_fc.weight[:old_n] = old_fc.weight
                new_fc.bias[:old_n]   = old_fc.bias
            self._head.net[-1] = new_fc

        logger.info("Fine-tuning on %d new examples for %d epochs …", len(records), epochs)
        X = self._encode(messages)
        Y = self._to_multihot(tool_lists)

        ds = ToolDataset(X, Y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self._head.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(1, epochs + 1):
            self._head.train()
            total = 0.0
            for Xb, Yb in dl:
                Xb, Yb = Xb.to(self.device), Yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._head(Xb), Yb)
                loss.backward()
                optimizer.step()
                total += loss.item()
            logger.info("Fine-tune epoch %d/%d  loss=%.4f", epoch, epochs, total / len(dl))

        logger.info("Fine-tuning done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Data collection helper  (call this inside run_turn_local to auto-log)
# ═══════════════════════════════════════════════════════════════════════════════

class ToolCallLogger:
    """
    Append tool-call events to a JSONL file for future training.

    Usage inside v1_colab_brain.py
    ───────────────────────────────
      from router_classifier import ToolCallLogger
      logger_tc = ToolCallLogger("tool_calls.jsonl")

      # inside run_turn_local(), after parsing tool calls:
      logger_tc.log(message=message, tools_called=[name])
    """

    def __init__(self, path: str | Path = "tool_calls.jsonl"):
        self.path = Path(path)

    def log(self, message: str, tools_called: list[str]) -> None:
        record = {"message": message, "tools": tools_called}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Evaluation report
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(rc: RouterClassifier, data_path: str | Path, threshold: Optional[float] = None) -> dict:
    """
    Compute per-tool precision / recall / F1 on a held-out JSONL file.
    Prints a formatted report and returns the metrics dict.
    """
    records = load_jsonl(data_path)
    messages  = [r["message"] for r in records]
    truth_raw = [r.get("tools", []) for r in records]

    predictions = rc.predict(messages, threshold=threshold)

    # Aggregate per-tool
    stats: dict[str, dict] = {}
    all_tools = set(rc.label_vocab)

    for tool in all_tools:
        tp = fp = fn = tn = 0
        for pred, truth in zip(predictions, truth_raw):
            p = tool in pred
            t = tool in truth
            if p and t:  tp += 1
            elif p:      fp += 1
            elif t:      fn += 1
            else:        tn += 1
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        stats[tool] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=round(prec, 3),
                           recall=round(rec, 3), f1=round(f1, 3), support=tp + fn)

    # Global micro-F1
    total_tp = sum(v["tp"] for v in stats.values())
    total_fp = sum(v["fp"] for v in stats.values())
    total_fn = sum(v["fn"] for v in stats.values())
    micro_p  = total_tp / (total_tp + total_fp + 1e-9)
    micro_r  = total_tp / (total_tp + total_fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    # Print report
    header = f"{'Tool':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}"
    print("\n" + "═" * 62)
    print("  ROUTER CLASSIFIER — EVALUATION REPORT")
    print("═" * 62)
    print(header)
    print("─" * 62)
    for tool, m in sorted(stats.items(), key=lambda x: -x[1]["f1"]):
        print(f"  {tool:<28} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")
    print("─" * 62)
    print(f"  {'MICRO AVG':<28} {micro_p:>6.3f} {micro_r:>6.3f} {micro_f1:>6.3f}")
    print("═" * 62 + "\n")

    stats["__micro__"] = dict(precision=round(micro_p, 3), recall=round(micro_r, 3), f1=round(micro_f1, 3))
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Synthetic data generator  (bootstrap when you have < 100 examples)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    tool_names: list[str],
    n_per_tool: int = 50,
    out_path: str | Path = "synthetic_tool_calls.jsonl",
) -> None:
    """
    Generate simple synthetic training examples (one per tool) using
    heuristic templates.  Good as a cold-start bootstrap; replace with real
    logged data as quickly as possible.

    tool_names : list of tool names your agent supports, e.g.
                 ['run_command', 'web_fetch', 'read_file', 'write_file', …]
    n_per_tool : how many examples to generate per tool
    out_path   : output JSONL path
    """
    import random
    random.seed(42)

    TEMPLATES = {
        "run_command": [
            "run {cmd} in the terminal",
            "execute {cmd}",
            "open a new terminal and {cmd}",
            "list files in the current directory",
            "kill the process {pid}",
            "show me system resource usage",
            "restart the service {svc}",
            "run ls -la",
            "check disk space",
            "ping google.com",
        ],
        "web_fetch": [
            "go to {url}",
            "open {url} in the browser",
            "search the web for {q}",
            "fetch the content of {url}",
            "what is on {url}",
            "browse to {url}",
            "download the webpage {url}",
        ],
        "read_file": [
            "read the file {file}",
            "show me the contents of {file}",
            "open {file}",
            "cat {file}",
            "display {file}",
            "what is in {file}",
        ],
        "write_file": [
            "write {content} to {file}",
            "save {content} in {file}",
            "create a new file called {file}",
            "append {content} to {file}",
        ],
        "search_web": [
            "search for {q}",
            "google {q}",
            "look up {q} online",
            "find information about {q}",
            "what is {q}",
        ],
    }

    FILL = {
        "{cmd}":     ["ls", "pwd", "htop", "df -h", "ps aux", "cat /etc/hosts", "echo hello"],
        "{url}":     ["github.com", "google.com", "stackoverflow.com", "docs.python.org"],
        "{q}":       ["Python list comprehension", "best GPU for ML", "transformer architecture", "linux commands"],
        "{file}":    ["config.json", "data.csv", "README.md", "model.pt", "notes.txt"],
        "{content}": ["hello world", "test data", "my name is Friday"],
        "{pid}":     ["1234", "5678"],
        "{svc}":     ["nginx", "postgresql", "redis"],
    }

    import random

    def fill(template: str) -> str:
        for key, choices in FILL.items():
            if key in template:
                template = template.replace(key, random.choice(choices), 1)
        return template

    with open(out_path, "w", encoding="utf-8") as f:
        for tool in tool_names:
            tpls = TEMPLATES.get(tool, [f"use {tool}", f"call {tool} now", f"invoke {tool}"])
            for _ in range(n_per_tool):
                msg = fill(random.choice(tpls))
                f.write(json.dumps({"message": msg, "tools": [tool]}) + "\n")

        # add some no-tool examples
        no_tool_msgs = [
            "what is 2 + 2",
            "tell me a joke",
            "who are you",
            "hello",
            "what can you do",
            "how are you",
        ]
        for msg in no_tool_msgs * (n_per_tool // len(no_tool_msgs) + 1):
            f.write(json.dumps({"message": msg, "tools": []}) + "\n")

    logger.info("Wrote synthetic data to %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Quick demo / smoke test  (run this file directly: python router_classifier.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RouterClassifier CLI")
    parser.add_argument("--data",      default="tool_calls.jsonl",      help="Training JSONL file")
    parser.add_argument("--save",      default="router.pt",              help="Save path")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--synth",     action="store_true",
                        help="Generate synthetic data first (cold start)")
    parser.add_argument("--eval",      default=None, help="JSONL file to evaluate on after training")
    parser.add_argument("--query",     default=None, help="Run a single inference after training")
    args = parser.parse_args()

    if args.synth:
        print("Generating synthetic bootstrap data …")
        generate_synthetic_data(
            tool_names=["run_command", "web_fetch", "read_file", "write_file", "search_web"],
            n_per_tool=80,
            out_path=args.data,
        )

    print(f"\n{'═'*60}")
    print("  TRACE Router Classifier")
    print(f"{'═'*60}\n")

    rc = RouterClassifier(threshold=args.threshold)
    history = rc.train(args.data, epochs=args.epochs)
    rc.save(args.save)

    # ── quick inference demo ──────────────────────────────────────────────────
    demo_queries = [
        "open chrome and navigate to github.com",
        "list all files in the current folder",
        "what is the capital of France",
        "write hello world to notes.txt",
        "search the internet for transformer models",
    ]
    if args.query:
        demo_queries = [args.query]

    print("\n── Inference Demo ──")
    for q in demo_queries:
        predicted = rc.predict(q)
        scores    = rc.predict_scores(q)
        top3 = list(scores.items())[:3]
        print(f"\n  Query   : {q}")
        print(f"  Predicted tools: {predicted}")
        print(f"  Top scores: {top3}")

    # ── eval report ──────────────────────────────────────────────────────────
    if args.eval:
        evaluate(rc, args.eval, threshold=args.threshold)

    print("\n✅ Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Colab integration snippet  (copy-paste into your colab notebook)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ── Cell A: Install + import ──────────────────────────────────────────────────
#
#   !pip install -q sentence-transformers
#   from router_classifier import RouterClassifier, ToolCallLogger, evaluate, generate_synthetic_data
#
# ── Cell B: Generate synthetic data (cold start) ─────────────────────────────
#
#   generate_synthetic_data(
#       tool_names=["run_command", "web_fetch", "read_file", "write_file"],
#       n_per_tool=100,
#       out_path="tool_calls.jsonl",
#   )
#
# ── Cell C: Train ─────────────────────────────────────────────────────────────
#
#   rc = RouterClassifier(threshold=0.45)
#   history = rc.train("tool_calls.jsonl", epochs=30)
#   rc.save("router.pt")
#
# ── Cell D: Evaluate ─────────────────────────────────────────────────────────
#
#   evaluate(rc, "tool_calls.jsonl")          # use val split in practice
#
# ── Cell E: Integrate into v1_colab_brain.py ─────────────────────────────────
#
#   rc = RouterClassifier.load("router.pt")
#   tc_logger = ToolCallLogger("tool_calls.jsonl")
#
#   # BEFORE calling _generate() in run_turn_local():
#   predicted_tools = rc.predict(message)
#   filtered_tools  = [t for t in qwen_tools if t["function"]["name"] in predicted_tools] or qwen_tools
#   output = _generate(messages, filtered_tools)   # ← faster & more accurate
#
#   # AFTER a tool call is confirmed:
#   tc_logger.log(message=message, tools_called=[name])   # ← grow your dataset
#
# ─────────────────────────────────────────────────────────────────────────────
