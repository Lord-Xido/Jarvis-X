"""SQLite persistent source metadata and exact cosine vector retrieval."""
import json
import sqlite3
from pathlib import Path
import numpy as np

DDL = '''
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS Messages (
 sample_id INTEGER PRIMARY KEY, text TEXT NOT NULL, label TEXT NOT NULL,
 split TEXT NOT NULL, archive_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS Media (
 id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER NOT NULL,
 modality TEXT NOT NULL, archive_path TEXT NOT NULL,
 FOREIGN KEY(sample_id) REFERENCES Messages(sample_id),
 UNIQUE(sample_id, modality)
);
CREATE TABLE IF NOT EXISTS Reactions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER NOT NULL,
 evaluation_tag TEXT NOT NULL, score REAL,
 FOREIGN KEY(sample_id) REFERENCES Messages(sample_id)
);
CREATE TABLE IF NOT EXISTS Embeddings (
 sample_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL,
 vector BLOB NOT NULL, norm REAL NOT NULL,
 FOREIGN KEY(sample_id) REFERENCES Messages(sample_id)
);
'''


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(DDL)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def add(self, sample_id, tokens, label, split, archive_path, embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError('invalid embedding')
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            raise ValueError('zero embedding')
        with self.db:
            self.db.execute('''INSERT OR REPLACE INTO Messages(sample_id,text,label,split,archive_path)
                               VALUES(?,?,?,?,?)''', (int(sample_id), json.dumps([int(i) for i in tokens]),
                                                  str(label), str(split), str(archive_path)))
            for m in ('image', 'audio', 'video'):
                self.db.execute('''INSERT OR REPLACE INTO Media(sample_id,modality,archive_path)
                                   VALUES(?,?,?)''', (int(sample_id), m, str(archive_path)))
            self.db.execute('''INSERT OR REPLACE INTO Embeddings(sample_id,dim,vector,norm)
                               VALUES(?,?,?,?)''', (int(sample_id), vector.size,
                                                    vector.tobytes(), norm))

    def react(self, sample_id, tag, score=None):
        with self.db:
            self.db.execute('INSERT INTO Reactions(sample_id,evaluation_tag,score) VALUES(?,?,?)',
                            (int(sample_id), str(tag), score))

    def top_k(self, vector, k=5, exclude_id=None, split=None):
        q = np.asarray(vector, dtype=np.float32).ravel()
        norm_q = float(np.linalg.norm(q))
        if not np.isfinite(q).all() or norm_q < 1e-12 or not 1 <= k <= 100:
            raise ValueError('invalid query or k')
        rows = self.db.execute('''SELECT e.sample_id,e.dim,e.vector,e.norm,m.label,m.split,m.archive_path
                                  FROM Embeddings e JOIN Messages m USING(sample_id)''').fetchall()
        out = []
        for sample_id, dim, payload, norm, label, split_row, archive_path in rows:
            if sample_id == exclude_id or (split is not None and split != split_row):
                continue
            if dim != q.size:
                raise ValueError('query vector dimension mismatch')
            v = np.frombuffer(payload, dtype=np.float32)
            score = float(np.dot(v, q) / (max(1e-12, norm) * norm_q))
            out.append({'sample_id': sample_id, 'similarity': score,
                        'label': label, 'split': split_row, 'archive_path': archive_path})
        return sorted(out, key=lambda e: (-e['similarity'], e['sample_id']))[:k]

    def embedding(self, sample_id):
        row = self.db.execute('SELECT dim,vector FROM Embeddings WHERE sample_id=?',
                              (int(sample_id),)).fetchone()
        if row is None:
            raise KeyError(f'no latent embedding for sample {sample_id}')
        dim, payload = row
        vec = np.frombuffer(payload, dtype=np.float32)
        if vec.size != dim:
            raise ValueError('corrupt embedding dimension')
        return vec.copy()

    def counts(self):
        return {t: self.db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                for t in ('Messages', 'Media', 'Reactions', 'Embeddings')}
