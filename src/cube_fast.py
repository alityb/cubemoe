"""Vectorised cube engine on 54-sticker arrays, plus whole-cube rotations for data augmentation.

Sticker layout = probe/cube.to_facelets order (URFDLB faces, 9 stickers each), colours as ints 0-5 in
'URFDLB' order (the encoding used by every dataset in this repo).

  MOVE_PERM[m]   : new[:, i] = old[:, MOVE_PERM[m][i]]  for move m in MOVE_NAMES order
  apply(states, m)               vectorised move application, states (N, 54)
  ROTATIONS                      the 8 whole-cube rotations that keep the U-D axis (so they map
                                 G1 = <U,D,R2,L2,F2,B2> onto itself and preserve the Kociemba phase)
  rotate(states, moves, r)       rotated states (colours relabelled so centres stay canonical) and the
                                 correspondingly renamed moves
Everything here is checked against probe/cube.py by self_test().
"""
import itertools, numpy as np

MOVE_NAMES = [f + s for f in 'URFDLB' for s in ('', "'", '2')]
FACES = 'URFDLB'
NORMAL = {'U': (0, 1, 0), 'R': (1, 0, 0), 'F': (0, 0, 1), 'D': (0, -1, 0), 'L': (-1, 0, 0), 'B': (0, 0, -1)}

def _facelet_pos():
    """3-D centre of every sticker; cube spans [-1, 1], stickers sit at +-1.5 on their face."""
    P = []
    for f in FACES:
        for r in range(3):
            for c in range(3):
                if f == 'U': p = (c - 1, 1.5, r - 1)        # seen from above, B at top
                elif f == 'R': p = (1.5, 1 - r, 1 - c)      # seen from right, U at top
                elif f == 'F': p = (c - 1, 1 - r, 1.5)
                elif f == 'D': p = (c - 1, -1.5, 1 - r)     # seen from below, F at top
                elif f == 'L': p = (-1.5, 1 - r, c - 1)
                else:          p = (1 - c, 1 - r, -1.5)     # B, seen from back, U at top
                P.append(p)
    return np.array(P, float)
POS = _facelet_pos()

def _index_of(points):
    out = []
    for q in points:
        d = np.abs(POS - q).sum(1); j = int(d.argmin()); assert d[j] < 1e-6, q; out.append(j)
    return np.array(out)

def _rot_matrix(axis, quarter_turns_cw):
    """Rotation by quarter_turns_cw * 90 deg CLOCKWISE when looking at the face whose normal is `axis`."""
    a = np.array(axis, float); ang = -np.pi / 2 * quarter_turns_cw
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.round(np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K).astype(int)

def _geometric_move_perm(name):
    f = name[0]; q = {'': 1, "'": 3, '2': 2}[name[1:]]
    Rm = _rot_matrix(NORMAL[f], q); n = np.array(NORMAL[f])
    perm = np.arange(54)
    layer = POS @ n > 0.5                                    # stickers in the turning layer
    src = np.flatnonzero(layer); dst = _index_of(POS[src] @ Rm.T)
    perm[dst] = src                                          # sticker at src moves to dst: new[dst] = old[src]
    return perm

MOVE_PERM = np.stack([_geometric_move_perm(m) for m in MOVE_NAMES])

def apply(states, m):
    return states[:, MOVE_PERM[m]]

SOLVED = np.repeat(np.arange(6), 9).astype(np.uint8)

def _all_rotations():
    mats = set(); out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), int)
            for i, (p, s) in enumerate(zip(perm, signs)): M[i, p] = s
            if round(np.linalg.det(M)) == 1 and M.tobytes() not in mats:
                mats.add(M.tobytes()); out.append(M)
    return out

def _rotation_tables(M):
    """For whole-cube rotation M: sticker permutation, colour relabelling, and move renaming."""
    dst = _index_of(POS @ M.T)                               # sticker j moves to position dst[j]
    perm = np.empty(54, int); perm[dst] = np.arange(54)      # new[i] = old[perm[i]]
    face_to = {}
    for f in FACES:
        img = tuple(int(x) for x in M @ np.array(NORMAL[f]))
        face_to[f] = [g for g in FACES if NORMAL[g] == img][0]
    # the colour that sat on face f's centre now sits on face face_to[f]; rename it so centres are canonical
    recolor = np.array([FACES.index(face_to[f]) for f in FACES], np.uint8)
    remap = np.array([MOVE_NAMES.index(face_to[m[0]] + m[1:]) for m in MOVE_NAMES])
    return perm, recolor, remap

_ALL = _all_rotations()
ROTATIONS = [M for M in _ALL if abs(int((M @ np.array([0, 1, 0]))[1])) == 1]   # keep the U-D axis
ROT_PERM, ROT_RECOLOR, ROT_MOVE = map(np.stack, zip(*[_rotation_tables(M) for M in ROTATIONS]))

def rotate(states, moves, r):
    return ROT_RECOLOR[r][states[:, ROT_PERM[r]]], ROT_MOVE[r][moves]

def self_test(n=300, seed=0):
    import sys; sys.path.insert(0, '/Users/alityb/projects/cubemoe/probe')
    import cube, random
    rng = random.Random(seed); CID = {c: i for i, c in enumerate(FACES)}
    enc = lambda s: np.array([CID[c] for c in cube.to_facelets(s)], np.uint8)
    S = [cube.random_state(rng) for _ in range(n)]; A = np.stack([enc(s) for s in S])
    assert (enc(cube.SOLVED) == SOLVED).all(), 'solved encoding'
    bad = 0
    for m, name in enumerate(MOVE_NAMES):
        bad += int((apply(A, m) != np.stack([enc(cube.apply_move(s, name)) for s in S])).any())
    print(f"moves: geometric sticker permutations agree with probe/cube.py on {18-bad}/18 moves x {n} random cubes")
    assert bad == 0
    print(f"rotations: {len(_ALL)} proper rotations, {len(ROTATIONS)} keep the U-D axis")
    ok_s = ok_c = ok_g = 0
    G1 = set(MOVE_NAMES.index(x) for x in ('U', "U'", 'U2', 'D', "D'", 'D2', 'R2', 'L2', 'F2', 'B2'))
    for r in range(len(ROTATIONS)):
        Z, _ = rotate(SOLVED[None], np.array([0]), r); ok_s += int((Z == SOLVED).all())
        good = True
        for m in range(18):
            lhs, _ = rotate(apply(A, m), np.array([m]), r)
            Ar, mr = rotate(A, np.array([m]), r)
            good &= bool((lhs == apply(Ar, int(mr[0]))).all())
        ok_c += int(good)
        ok_g += int(set(int(ROT_MOVE[r][g]) for g in G1) == G1)
    print(f"rotations keep the solved cube solved: {ok_s}/8; commute with every move (rot(move(s)) == "
          f"move'(rot(s))): {ok_c}/8; map the G1 move set onto itself: {ok_g}/8")
    from qa import facelets_to_cubie
    agree = 0
    for s in S[:100]:
        for r in range(8):
            Z, _ = rotate(enc(s)[None], np.array([0]), r)
            agree += int(cube.in_G1(facelets_to_cubie(''.join(FACES[c] for c in Z[0]))) == cube.in_G1(s))
    print(f"G1 membership preserved under all 8 rotations: {agree}/800")
    assert ok_s == 8 and ok_c == 8 and ok_g == 8 and agree == 800

if __name__ == '__main__':
    self_test()
