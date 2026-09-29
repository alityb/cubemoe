# Artifact manifest

Generated 2026-09-29. Heavy artifacts are **gitignored** (GitHub 100 MB/file limit; checkpoints in git once bloated the repo to 2 GB). This file records every one so they can be restored, **verified by sha256**, or regenerated. Verify a restored copy with `shasum -a 256 <file>`.

Total: 138 files, 3.89 GB.

Not in this list but needed: Hugging Face model weights (re-downloadable): `allenai/OLMoE-1B-7B-0924`, `ibm-granite/granite-3.1-1b-a400m-base`. `llm/corpus.json` IS in git (machine-dependent: built from this machine's Python stdlib and installed packages, so it cannot be rebuilt identically elsewhere).

| file | MB | sha256 | produced by | if lost |
|---|---|---|---|---|
| `out/V1_top1_e8_local_s30.pt` | 107.5 | `cd32d11e52a0d826…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V1_top1_e8_local_s31.pt` | 107.5 | `2c76370a29a093a4…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V1_top1_e8_local_s32.pt` | 107.5 | `87d31a6a2bfea2db…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V1_top1_e8_local_s33.pt` | 107.5 | `7f4250305c0cbd04…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V1_top1_e8_local_s34.pt` | 107.5 | `99d33a904b0a82b0…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V6_top1_e8_large_s30.pt` | 107.5 | `42f32cdff9b8868e…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V6_top1_e8_large_s31.pt` | 107.5 | `ae04e7e04ddafa15…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V6_top1_e8_large_s32.pt` | 107.5 | `4d9f7a84235ead65…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V6_top1_e8_large_s33.pt` | 107.5 | `5e92a467e1e2ec34…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/V6_top1_e8_large_s34.pt` | 107.5 | `dccf1498fe265768…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/dense_mixed.pt` | 5.4 | `09b5f56ca9f0279a…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/dense_mixed_s1.pt` | 5.4 | `12a2d274929cc1ea…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/dense_mixed_s2.pt` | 5.4 | `39ed4f8f3935d63b…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/dense_mixed_s3.pt` | 5.4 | `b9375119d969445c…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/dense_mixed_s4.pt` | 5.4 | `e0d518a3aa50701a…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfop_s40.pt` | 18.2 | `04c7f71153596b68…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfop_s41.pt` | 18.2 | `5b29a9b91cfb4cbb…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfop_s42.pt` | 18.2 | `0ac202a3efd063fc…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfop_s43.pt` | 18.2 | `ba56e4a53db00dfc…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfop_s44.pt` | 18.2 | `a82cc54b00af7e67…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfopdec_s50.pt` | 18.2 | `21c42a902096818d…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfopdec_s51.pt` | 18.2 | `fa2cfefd3817083d…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfopdec_s52.pt` | 18.2 | `6c2f14afce00f976…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfopdec_s53.pt` | 18.2 | `c71c82e6c26f45ac…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_cfopdec_s54.pt` | 18.2 | `902101b954e1b2f2…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_mixed.pt` | 18.2 | `fc47809ed17c8fd8…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_mixed_s10.pt` | 18.2 | `86c6b3b7ca3311f2…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_mixed_s11.pt` | 18.2 | `113705775abc618b…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_mixed_s12.pt` | 18.2 | `b9cd36437f0164c2…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_mixed_s13.pt` | 18.2 | `ef7f3dd896d1e5a5…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_mixed_s14.pt` | 18.2 | `147093f988f62421…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/hash_mixed_s20.pt` | 107.7 | `3ff5b92523162b4b…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/hash_mixed_s21.pt` | 107.7 | `b857e06daa29b83d…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/hash_mixed_s22.pt` | 107.7 | `2f18f98ed1071f29…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/hash_mixed_s23.pt` | 107.7 | `2d1c495e1320f660…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/hash_mixed_s24.pt` | 107.7 | `074f9ef68ac8667b…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/moe_cfop_s40.pt` | 18.0 | `96f90d1cc4a7e869…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfop_s41.pt` | 18.0 | `37b5cd84d5cbfa69…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfop_s42.pt` | 18.0 | `4f60422cd4511447…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfop_s43.pt` | 18.0 | `389918254c278f18…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfop_s44.pt` | 18.0 | `5fe53d569abf2762…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfopdec_s50.pt` | 18.1 | `951d8185908f3268…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfopdec_s51.pt` | 18.1 | `1fb8dd7945e1b4f8…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfopdec_s52.pt` | 18.1 | `8a6e9d62b6b1e846…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfopdec_s53.pt` | 18.1 | `5a07a9e71c220417…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_cfopdec_s54.pt` | 18.1 | `bda6f7bc73fc5c11…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed.pt` | 18.0 | `8220f9e9ec527ff3…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s1.pt` | 18.0 | `a9ec56e1261d6074…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s10.pt` | 18.0 | `a8bf06f8b3927652…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s11.pt` | 18.0 | `ec38274549b073d1…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s12.pt` | 18.0 | `a9942e3903f66d69…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s13.pt` | 18.0 | `db3d0552c97cffbe…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s14.pt` | 18.0 | `3f87738c5a142ba1…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s2.pt` | 18.0 | `e97ce7b1657354f4…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/moe_mixed_s20.pt` | 107.5 | `f09abc1c6e307470…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/moe_mixed_s21.pt` | 107.5 | `8f0c09c5239adb22…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/moe_mixed_s22.pt` | 107.5 | `e054db6d1e0dfa12…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/moe_mixed_s23.pt` | 107.5 | `ce264849caf217ef…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/moe_mixed_s24.pt` | 107.5 | `55c011e8a179ce85…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/moe_mixed_s3.pt` | 18.0 | `905f65fb0dbbb50f…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_mixed_s4.pt` | 18.0 | `7f6e38386455a009…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/moe_naive.pt` | 18.0 | `4513aeca9bdcfd64…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_cfop_s60.pt` | 18.0 | `36b50c81dedf6efc…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_cfop_s61.pt` | 18.0 | `e69fd1d0f0cc58ea…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_cfop_s62.pt` | 18.0 | `bd6be3cc13f66aff…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_cfop_s63.pt` | 18.0 | `03645194f73cd168…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_cfop_s64.pt` | 18.0 | `d7e2eea78e5852f6…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_dense_mixed.pt` | 5.3 | `ac43fb0dc7c5a515…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_dense_mixed_s1.pt` | 5.3 | `4b1c90e7821f685a…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_dense_mixed_s2.pt` | 5.3 | `2fcfa9da15135fa5…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed.pt` | 18.0 | `ed983275e6d230b9…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s1.pt` | 18.0 | `6d2e849cfd79754d…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s10.pt` | 18.0 | `6d0184bc477ade05…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s11.pt` | 18.0 | `81249a5b82c84d55…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s12.pt` | 18.0 | `b24e327540a9093a…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s13.pt` | 18.0 | `87ad72ab46af794e…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s14.pt` | 18.0 | `236d3d7d1840a544…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s2.pt` | 18.0 | `68878212bf4800b3…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/state_mixed_s20.pt` | 107.4 | `7deed39ef216ccb4…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/state_mixed_s21.pt` | 107.4 | `7baf4c19e113de6c…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/state_mixed_s22.pt` | 107.4 | `45aec15de3892e50…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/state_mixed_s23.pt` | 107.4 | `bde7898421b30619…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/state_mixed_s24.pt` | 107.4 | `cd02e9f12d88600f…` | src/train.py on Modal CUDA (6L/d256, 250k) | **costly: paid GPU** |
| `out/state_mixed_s3.pt` | 18.0 | `698eed5148a0c935…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s4.pt` | 18.0 | `40f0e1ed25ed785b…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `out/state_mixed_s5.pt` | 18.0 | `fd20e27bf0318909…` | src/train.py (local MPS or Modal) | regenerable (~8-50 min each) |
| `data/cfop.npz` | 8.2 | `010f7c5a1e9fb1a5…` | src/gen_data.py / src/gen_cfop.py (seeded) | regenerable (minutes-hours) |
| `data/cfop_dec.npz` | 9.6 | `a6758f501fbd16cf…` | src/gen_data.py / src/gen_cfop.py (seeded) | regenerable (minutes-hours) |
| `data/forced.npz` | 12.3 | `d4aeb90201f9a6ea…` | src/gen_data.py / src/gen_cfop.py (seeded) | regenerable (minutes-hours) |
| `data/mixed.npz` | 8.6 | `9d352e86ba5fdd81…` | src/gen_data.py / src/gen_cfop.py (seeded) | regenerable (minutes-hours) |
| `data/mixed250k.npz` | 86.1 | `80ea630325a31bea…` | src/gen_data.py / src/gen_cfop.py (seeded) | regenerable (minutes-hours) |
| `data/naive.npz` | 8.9 | `2928360f8805c3d6…` | src/gen_data.py / src/gen_cfop.py (seeded) | regenerable (minutes-hours) |
| `out/explore/cfopfeat_cfop.npz` | 0.1 | `4ccc7a3bd846ddce…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfopfeat_cfop_dec.npz` | 0.1 | `480b945b890a850f…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfopmeta_cfop.npz` | 0.3 | `777963b065de6aa4…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfopmeta_cfop_dec.npz` | 0.3 | `86b288bfb45f9b48…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfop_s40.npz` | 0.1 | `bcfcc2deeec3a082…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfop_s41.npz` | 0.1 | `968663b62682c6a5…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfop_s42.npz` | 0.1 | `e3ea8b6e124c4bb4…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfop_s43.npz` | 0.1 | `842f4fa9e7ae6685…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfop_s44.npz` | 0.1 | `2ac010e08f836232…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfopdec_s50.npz` | 0.1 | `2f96b2571854965f…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfopdec_s51.npz` | 0.1 | `25786ecf2b3c5c08…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfopdec_s52.npz` | 0.1 | `372b2c8e2392e829…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfopdec_s53.npz` | 0.1 | `dd8f99482e167ba4…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_hash_cfopdec_s54.npz` | 0.1 | `334e798d147195ed…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfop_s40.npz` | 0.1 | `19ca408866220201…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfop_s41.npz` | 0.1 | `d50d6f2a99cecfb9…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfop_s42.npz` | 0.1 | `474b9af5e6acf8c0…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfop_s43.npz` | 0.1 | `bc7ce1db216feefa…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfop_s44.npz` | 0.1 | `2fd498057548d811…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfopdec_s50.npz` | 0.1 | `00f6bac580ace8b9…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfopdec_s51.npz` | 0.1 | `3192aa1e5eb4dbe4…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfopdec_s52.npz` | 0.1 | `acb017460543e06c…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfopdec_s53.npz` | 0.1 | `de0cc356fa450bc7…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/cfoproute_moe_cfopdec_s54.npz` | 0.1 | `c266b636d17a4788…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s10.npz` | 0.2 | `73a4880479982e0d…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s11.npz` | 0.2 | `6b0be491da56b56c…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s12.npz` | 0.2 | `b38d182cc8061e41…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s13.npz` | 0.2 | `5f755fc6d45184cc…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s14.npz` | 0.2 | `f6d5a774d9d3a071…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s20.npz` | 0.3 | `ac2635b627f0553c…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s21.npz` | 0.3 | `8775d15cd976d9a5…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s22.npz` | 0.3 | `ff917214dc27466a…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s23.npz` | 0.3 | `39400eae2c800088…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `out/explore/g1_moe_mixed_s24.npz` | 0.3 | `82c9f123d496eb77…` | src/explore_*.py | regenerable from checkpoints (minutes) |
| `llm/base_OLMoE-1B-7B-0924.npz` | 2.1 | `f2192e743ef19d09…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/base_granite-3.1-1b-a400m-base.npz` | 2.5 | `1116d329f6b9e371…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_OLMoE-1B-7B-0924_L11.npz` | 9.1 | `6e1008f8811ee815…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_OLMoE-1B-7B-0924_L15.npz` | 9.1 | `eb71c4e101f0bc30…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_OLMoE-1B-7B-0924_L3.npz` | 9.3 | `1dad29128974497c…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_OLMoE-1B-7B-0924_L7.npz` | 9.2 | `972567931855a62d…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_granite-3.1-1b-a400m-base_L11.npz` | 4.6 | `e903bc6675899a96…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_granite-3.1-1b-a400m-base_L15.npz` | 4.6 | `3e4b0106fa68cb06…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_granite-3.1-1b-a400m-base_L19.npz` | 4.6 | `7cfbae871592a90b…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_granite-3.1-1b-a400m-base_L23.npz` | 1.5 | `16a0ce498bed23f5…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_granite-3.1-1b-a400m-base_L3.npz` | 4.6 | `2944d0faedb8390c…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |
| `llm/ko_granite-3.1-1b-a400m-base_L7.npz` | 4.6 | `a524931793b22992…` | llm/pilot.py (base / ko stages) | **costly: up to ~5 h compute** |

## Full sha256 list

```
cd32d11e52a0d8267d0745a10a96ba46895cd5fb7a3e2867508e12f98dee7da6  out/V1_top1_e8_local_s30.pt
2c76370a29a093a425dea6e71368f92e6637d435b4c3cc6b1708d0cb2b801733  out/V1_top1_e8_local_s31.pt
87d31a6a2bfea2dbe543330901daeac5883a0f8c67d5b20d808b5efa4190bac2  out/V1_top1_e8_local_s32.pt
7f4250305c0cbd04b9d510b5e4c881f3550768b7b5ed2aef32f5fdf412f52664  out/V1_top1_e8_local_s33.pt
99d33a904b0a82b0e7d709665b0ddcdeb60c452e369cbc87e5d8b8f4a89519a6  out/V1_top1_e8_local_s34.pt
42f32cdff9b8868eab1a529a3bf592c72b052f17a885777b002a5c0de64c8125  out/V6_top1_e8_large_s30.pt
ae04e7e04ddafa153f1fc9622137cdda2781ef653fc7b1dff7d743a9a890d77a  out/V6_top1_e8_large_s31.pt
4d9f7a84235ead65a6a84a3849fcf8119c976ff45500ae563a7f6db1470e7846  out/V6_top1_e8_large_s32.pt
5e92a467e1e2ec340a3cbbee961161f9a804587ddd0544201d32f991c9fe0b46  out/V6_top1_e8_large_s33.pt
dccf1498fe26576844693306d51018b3ea3425baf78e92cf8b1c6f8923825b47  out/V6_top1_e8_large_s34.pt
09b5f56ca9f0279a657610cba357f70d1c02ff072c40481c47258983a0308cfa  out/dense_mixed.pt
12a2d274929cc1eaf897b0a9dd8c3cb98d38e6f6a42eaa09d0fca2720b6d4d04  out/dense_mixed_s1.pt
39ed4f8f3935d63ba41654c07b2ee8a717d624bf222e3f52cdea5a6f67f14b54  out/dense_mixed_s2.pt
b9375119d969445c5554efe1201739fcb198a63b422336cca66ce50b7184598d  out/dense_mixed_s3.pt
e0d518a3aa50701a5b091f07de83cf4021fa364d73eef92129e9117d420778dd  out/dense_mixed_s4.pt
04c7f71153596b68e015097c2ee3bf0a28dd8cee1dfed84d0953df1517539480  out/hash_cfop_s40.pt
5b29a9b91cfb4cbba8e2a7d16c4de367fcdb234a1076deb3a6e4a5314d0c94e7  out/hash_cfop_s41.pt
0ac202a3efd063fc97f523117a2cba76b0149949f4d5e6e3c2981a173390ad36  out/hash_cfop_s42.pt
ba56e4a53db00dfce24f2951fec641fe73761d762a6891daba282b733a7050cd  out/hash_cfop_s43.pt
a82cc54b00af7e67a3a27c4de8ae421155d5a70426b3b17cf3b9992805ee874b  out/hash_cfop_s44.pt
21c42a902096818dac0a5d59bfe3baac04b9e0608891e69b98180a69deacdac9  out/hash_cfopdec_s50.pt
fa2cfefd3817083d5d3b5134d552bf75a4b17dbb65aad38365c366093983aecc  out/hash_cfopdec_s51.pt
6c2f14afce00f976d5725508bc75158cb86b3160f033a2a380489e727337d908  out/hash_cfopdec_s52.pt
c71c82e6c26f45acb6d4479c344c9ab6701f273c925932386b4bc52c72f384df  out/hash_cfopdec_s53.pt
902101b954e1b2f2cdf973837891fa4f43a7ca4185c1914071eb188fbcefdc01  out/hash_cfopdec_s54.pt
fc47809ed17c8fd8ea2e5f9f76aef867f2f5dcb4224115aa896aa0072e3b48a9  out/hash_mixed.pt
86c6b3b7ca3311f22abc22310c0a1394139095fe8b589b253e1d60b89291f4a1  out/hash_mixed_s10.pt
113705775abc618bf6773b220f5e08b35e1585ee49fdf648c403d7c40282a33c  out/hash_mixed_s11.pt
b9cd36437f0164c2448f50e0f8a945f68eb5a968fd4fdf083108ffdedbcaafa4  out/hash_mixed_s12.pt
ef7f3dd896d1e5a528fa37d10d4ff0c8d8ca5aa79f06aa9e8d053a91b69a601c  out/hash_mixed_s13.pt
147093f988f62421873b84df95deec8507eb06d73ce20d04f7e5889e496194c2  out/hash_mixed_s14.pt
3ff5b92523162b4b9d7dc567ab03b7454dbaa201f7a08369fa9335569e7a948d  out/hash_mixed_s20.pt
b857e06daa29b83da7ceee6a62eac8e67eb085efb8b5a175cba5e2b7426ce9bb  out/hash_mixed_s21.pt
2f18f98ed1071f29379b5af945615a33c25e89fd0e96e73e94b42d66e84364eb  out/hash_mixed_s22.pt
2d1c495e1320f6608d9cec8b1ef21c11e3f806a17940e025064d7aa36b0f6ee3  out/hash_mixed_s23.pt
074f9ef68ac8667b98b3a3017d09cd34653b76418875df794446903910093ffc  out/hash_mixed_s24.pt
96f90d1cc4a7e869b4fb503f69fb76806684fc7954b94d970edabb92b018c81b  out/moe_cfop_s40.pt
37b5cd84d5cbfa691c851bc0f2b281a21d2082aaa81ea1e88ec65c3c6f2f21f6  out/moe_cfop_s41.pt
4f60422cd45114472a3cc6ec3401a660b734480dadaf6fc81cc486b5530674ea  out/moe_cfop_s42.pt
389918254c278f1878d0265460061d0c240f29c1830bdc53b654fd41cfc41374  out/moe_cfop_s43.pt
5fe53d569abf27622285fa5f5d9cfe8efc1882532b079821cd6d82a215745ed4  out/moe_cfop_s44.pt
951d8185908f32681956317d6a0e2934f2c266c47278a3070c087ceec01b20dc  out/moe_cfopdec_s50.pt
1fb8dd7945e1b4f8e4194b4644e34cf0d850a434e22c26930d4a194c439e7b51  out/moe_cfopdec_s51.pt
8a6e9d62b6b1e846499eb95cd88e19cd5535b7ff39612ded6fcbf8449392df28  out/moe_cfopdec_s52.pt
5a07a9e71c220417a312de21c1ae255cce45a88913591790c18dd5aacdba3df0  out/moe_cfopdec_s53.pt
bda6f7bc73fc5c113154b616d99db09be664167336477ae609513bd1d37d2f7b  out/moe_cfopdec_s54.pt
8220f9e9ec527ff32bba1032bc567cb842ed3810d677e701fd50276121ddc2ed  out/moe_mixed.pt
a9ec56e1261d6074145b808b850196495898ba679393c4570a0a11307fc8308c  out/moe_mixed_s1.pt
a8bf06f8b3927652656208b86c8939fa14921ad8ffd254a6e4a6a16e505f9797  out/moe_mixed_s10.pt
ec38274549b073d131f6c4074dff02b5cd98f66640fa8176b2e24b15cc53efa4  out/moe_mixed_s11.pt
a9942e3903f66d692ee15b28ecdea26542b5066deb3b3008c9830ef97d897cc9  out/moe_mixed_s12.pt
db3d0552c97cffbebd99d89d291f9beb2149ec0eb4e20ce4e6989c453fcc9186  out/moe_mixed_s13.pt
3f87738c5a142ba1614fb24341d947340d8c8dc52aef700573f4c4a55b90c9b4  out/moe_mixed_s14.pt
e97ce7b1657354f4d16b5212eefd6062c26e04d614710edbb4c13f84f15d4f50  out/moe_mixed_s2.pt
f09abc1c6e307470244f7cea4e3dae601a9fdb41b156b809d014f20beb16f332  out/moe_mixed_s20.pt
8f0c09c5239adb2251771bc754612fc20746aa96a5df4a77b600d81c47294b8f  out/moe_mixed_s21.pt
e054db6d1e0dfa1253c855386d63aecce1849144d8b8eb76f5dadecfac3a5c2c  out/moe_mixed_s22.pt
ce264849caf217ef7c1f26a6b3aa6bca31a344d92341913de7af0a6d659ecdaa  out/moe_mixed_s23.pt
55c011e8a179ce85a8b23aaf144564528a040a92869c55f6cbbc214329eb2e0c  out/moe_mixed_s24.pt
905f65fb0dbbb50f613d79d54453bb358adec0bbd336d2b652d9f602633f834b  out/moe_mixed_s3.pt
7f6e38386455a0094950e497d3964a9c0598a767c8ce69096b2c6eb1cd6ec88c  out/moe_mixed_s4.pt
4513aeca9bdcfd64aee4888e9091d7116e3f73146581babb93a08ab446b41bca  out/moe_naive.pt
36b50c81dedf6efc978663846d435b6548077413b7bc56f5b5a01c89686f7bc5  out/state_cfop_s60.pt
e69fd1d0f0cc58ea6fdacf624ba8ec1e40dd025d0ae264ff21309411b45bd881  out/state_cfop_s61.pt
bd6be3cc13f66affddb76490f1e82a736ef526bfc1052b9161ac9a9e842bdc76  out/state_cfop_s62.pt
03645194f73cd168322e3e584263902f5eefe04cd26119e1e85cdab332da4f0a  out/state_cfop_s63.pt
d7e2eea78e5852f66a787e3552b6f062a3e9bcf77de72301583933c9e5073d87  out/state_cfop_s64.pt
ac43fb0dc7c5a5157f77dbedc2f9837cad496dff4f824cd8adca6099d5455fd2  out/state_dense_mixed.pt
4b1c90e7821f685aca4a8bf810a1a00f82ea77fe2da53087212b6497d5797cc5  out/state_dense_mixed_s1.pt
2fcfa9da15135fa520e4bbaf46882bfec68a12d473f77d9fd38056acdf6decc5  out/state_dense_mixed_s2.pt
ed983275e6d230b941cc18275aee471651153be2ed0c1d4621eec57ff23d94a5  out/state_mixed.pt
6d2e849cfd79754d1aa2098e4652e61e26739d8ff0234e6e3005b2d44eea6b1e  out/state_mixed_s1.pt
6d0184bc477ade05264f387bb18508600e3f7c7eb41d1e90364222ed9bc80aa3  out/state_mixed_s10.pt
81249a5b82c84d55fc3cd9761ac473c553c2d783e326fc631366083643d3c982  out/state_mixed_s11.pt
b24e327540a9093af82628fe064c2f03b78f62d0dc47083c53f392f220eb16a6  out/state_mixed_s12.pt
87ad72ab46af794e1145d43127848c8c1d1b978ada2e3cb03b0eed65f3fbfaec  out/state_mixed_s13.pt
236d3d7d1840a544639f294ee49dfc96dc49836025ceeb6826663cad84b88494  out/state_mixed_s14.pt
68878212bf4800b3ca24e24644b5dffad1463fff1457ff8ec047e25c8158dd6a  out/state_mixed_s2.pt
7deed39ef216ccb4031c13559cc3da8b7b6646754c26ca4960e48b2323add75a  out/state_mixed_s20.pt
7baf4c19e113de6c4a5a8e6f803fd05f1007c5ac7085679dc3cdd9a33683dd10  out/state_mixed_s21.pt
45aec15de3892e5022cb955d5cbd6400126b3b7fd1b5d76307994628b96d717d  out/state_mixed_s22.pt
bde7898421b306194b3dda940cbf7d584a28324de0dcd8aaa7ae318477e2d758  out/state_mixed_s23.pt
cd02e9f12d88600f89266179e30c00b4de4f580edc0c7f9812919bb9661cd2b2  out/state_mixed_s24.pt
698eed5148a0c935d671c269643f267ba5a9f04512b18e4cf16ee1ca3f8ec34c  out/state_mixed_s3.pt
40f0e1ed25ed785b7b1151a15c9697dc1cd60f274896928fb50cb9ec9892c2af  out/state_mixed_s4.pt
fd20e27bf0318909bb095443fcbd511d50040def6ff84698a864343dac077e8c  out/state_mixed_s5.pt
010f7c5a1e9fb1a5259360429ff0a6012bedcfab596daacd529f617dcf12f8ae  data/cfop.npz
a6758f501fbd16cfa5403243492e356d00eaaa49d278b67d0a50f90061e0e365  data/cfop_dec.npz
d4aeb90201f9a6eaf29f4a692ff1b7c306c6e31917d03c9958d3b20d1d2abb14  data/forced.npz
9d352e86ba5fdd81bf1c742805d1d60feda186e623a5f08f44b2f5d5532c3972  data/mixed.npz
80ea630325a31bea106c7ef7be45b7cc881f748174be8436201efbc71e9b5a3d  data/mixed250k.npz
2928360f8805c3d64e3282420f8f35bd8579d6838282df9aa02487c84d47e68a  data/naive.npz
4ccc7a3bd846ddcebf1fc7059436bf5af82eef919730c7431a803171a7c0d5c8  out/explore/cfopfeat_cfop.npz
480b945b890a850f4e56cfc00f95467249b5ae3af9fa2da898e858d90654ec02  out/explore/cfopfeat_cfop_dec.npz
777963b065de6aa49942d0751c001337d287d9961e7678d5d12e040468ee5c62  out/explore/cfopmeta_cfop.npz
86b288bfb45f9b486176edf2eec88917d7466a5a049fba2bc5cf1f72520f24ad  out/explore/cfopmeta_cfop_dec.npz
bcfcc2deeec3a0826034b741ed07c42340c1f3f9588f9dff8bbff347fdc75425  out/explore/cfoproute_hash_cfop_s40.npz
968663b62682c6a54069218c24ae902500e76e940b0067900a505982596862b4  out/explore/cfoproute_hash_cfop_s41.npz
e3ea8b6e124c4bb4058bfa4e9c81228d875ad5c023f63d1406620859038959b5  out/explore/cfoproute_hash_cfop_s42.npz
842f4fa9e7ae6685a295bad3ad67195cb04961e8e126f5e6f53c0a003a0e476e  out/explore/cfoproute_hash_cfop_s43.npz
2ac010e08f836232498a7ff30c8d65fa70bcf6d6633bb7a75a74d1339f05967b  out/explore/cfoproute_hash_cfop_s44.npz
2f96b2571854965ff2341fb02f66bc87436f5eac011f15cef4337bfc111b6dca  out/explore/cfoproute_hash_cfopdec_s50.npz
25786ecf2b3c5c0802e60fb501d558b7977f72d07f992c3a13444ea8f70be478  out/explore/cfoproute_hash_cfopdec_s51.npz
372b2c8e2392e8292b308324b197a37ea4986743a61589f95abaaa90a8fd157e  out/explore/cfoproute_hash_cfopdec_s52.npz
dd8f99482e167ba46e2ecd72ccd8a43aa3bffa92d64ea1aabd78667cf0dfa286  out/explore/cfoproute_hash_cfopdec_s53.npz
334e798d147195ed9fb97b5e41360c72567e18bdfb6cc2f1bdf20fe81f963f5c  out/explore/cfoproute_hash_cfopdec_s54.npz
19ca408866220201d98200c3f804ca1c99ebe95e541fdc3a8d14de0d763b65d7  out/explore/cfoproute_moe_cfop_s40.npz
d50d6f2a99cecfb9c7d1896bb665ead86dabc28add69d48089c67dddbee9fb4a  out/explore/cfoproute_moe_cfop_s41.npz
474b9af5e6acf8c0e5479fc06aa9bec5b5e25688428b3db73d7e5c967b16a4e6  out/explore/cfoproute_moe_cfop_s42.npz
bc7ce1db216feefa85f9935f99aebf6b1802cc3ea554e6ca7ed089e42383b16a  out/explore/cfoproute_moe_cfop_s43.npz
2fd498057548d811d58ebf51a82db55d7506021ffca17376f9ee47630d1482ef  out/explore/cfoproute_moe_cfop_s44.npz
00f6bac580ace8b9b11cf136b675019334e38371c3093d2045460a52fd03a1a8  out/explore/cfoproute_moe_cfopdec_s50.npz
3192aa1e5eb4dbe4a5742547ad9a8220f05a81cb2ba1db544dd027347a28cb95  out/explore/cfoproute_moe_cfopdec_s51.npz
acb017460543e06c78ea5728ca822a243d855fd0f80de32d58d9246d298f8955  out/explore/cfoproute_moe_cfopdec_s52.npz
de0cc356fa450bc78278ffecad227cdef81d33204cbc56643d27eb2bc633f0b9  out/explore/cfoproute_moe_cfopdec_s53.npz
c266b636d17a4788d7db34b3c56fae7c636cb6ab3284e867354dd0d9e1f75274  out/explore/cfoproute_moe_cfopdec_s54.npz
73a4880479982e0d608fd5d9d88701c8b796ce2b304c4c1976bf66257316b74b  out/explore/g1_moe_mixed_s10.npz
6b0be491da56b56c8597226161b03d91558957b6dcf4eeeecd75c07a68a12a5b  out/explore/g1_moe_mixed_s11.npz
b38d182cc8061e410c4dc99008170cd904e1ec6b8edd14c5627890dde01fd946  out/explore/g1_moe_mixed_s12.npz
5f755fc6d45184cc5763a746cd02082334dfbd61428e6c69d1e329dbc9ff2c72  out/explore/g1_moe_mixed_s13.npz
f6d5a774d9d3a0714e181b790eb49ae33d6294794a84749006d9f095ee7f7651  out/explore/g1_moe_mixed_s14.npz
ac2635b627f0553cd9eb6e86b57884cbcfe171e1e07db788c1f37ab64fd7fed2  out/explore/g1_moe_mixed_s20.npz
8775d15cd976d9a5d2c246b90cbb19b9e3c402a85d0158d62a4a0b1b246dbda7  out/explore/g1_moe_mixed_s21.npz
ff917214dc27466a6f2d7bc0a7b7778cb90d1a4ffe19d5af0c2260542b669778  out/explore/g1_moe_mixed_s22.npz
39400eae2c8000889cc36b3741c386103b2c49f30f121d7701857df9151737c6  out/explore/g1_moe_mixed_s23.npz
82c9f123d496eb773f17e8894e59a688633a9a5103b3bdcebde4cbaaa6a54ac2  out/explore/g1_moe_mixed_s24.npz
f2192e743ef19d0922adb846638b6cb4afd32da075a15cf406f8361a3843418b  llm/base_OLMoE-1B-7B-0924.npz
1116d329f6b9e3710e9eff9b0ef88617155c7df9899dc30b7f538f78c8b059dd  llm/base_granite-3.1-1b-a400m-base.npz
6e1008f8811ee8156e5c6bbb0e73d1ac6d9c7b664887c4938e6df4c057b6a5f7  llm/ko_OLMoE-1B-7B-0924_L11.npz
eb71c4e101f0bc304da6cb8f1b7f1529ccab36b2fa5e513773d94bd91e5c3e99  llm/ko_OLMoE-1B-7B-0924_L15.npz
1dad29128974497c2d9afda2e1017cfe13bfc114229800c1785fef4e50656b01  llm/ko_OLMoE-1B-7B-0924_L3.npz
972567931855a62d7ffed338586fbd66350a645d958f850dbce86c64a2f4f9a6  llm/ko_OLMoE-1B-7B-0924_L7.npz
e903bc6675899a961522b0ec455798b9436f9ad103ef1d5c26f9c55acc99cecd  llm/ko_granite-3.1-1b-a400m-base_L11.npz
3e4b0106fa68cb0628dcc5a4f5eec826af149a7488a31ae85a78a14719eb4d67  llm/ko_granite-3.1-1b-a400m-base_L15.npz
7cfbae871592a90bd9ac021874aa8320a5403266d780b9e3b9dd7b255bfe31b7  llm/ko_granite-3.1-1b-a400m-base_L19.npz
16a0ce498bed23f558861db21ac525089b6c7598bf9adc1fc78297a9dbe92eed  llm/ko_granite-3.1-1b-a400m-base_L23.npz
2944d0faedb8390c00137d63aff47a4eaf29155f6a9357b10cc4ba19eda7bc3e  llm/ko_granite-3.1-1b-a400m-base_L3.npz
a524931793b2299226ae7c726f4d691595f826b6f363bb2917cdb4df9dad34f8  llm/ko_granite-3.1-1b-a400m-base_L7.npz
```
