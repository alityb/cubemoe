"""Applies the rules of prereg/PREREG_PATH_SWAP.md (#14) to $SWAP_OUT/*.json (default out/path_swap).
Usage: python src/path_swap_report.py  -> prints tables and the verdict; writes <out>/report.json"""
import json, os
R = '/Users/alityb/projects/cubemoe'; OUT = os.environ.get('SWAP_OUT', f'{R}/out/path_swap')
K = [f'mixtral_6L256_s{s}__mixed250k' for s in range(5)]; C = [f'cfop_mixtral_6L256_s{s}__cfop250k' for s in range(8)]

def main():
    O = {t: json.load(open(f'{OUT}/{t}.json')) for t in K + C}
    g = lambda t, s, k: O[t]['sets'][s][k]; lo = lambda t, s, k: g(t, s, k)['ci'][0]
    S0 = [t for t in K if g(t, 'all', 'changed')['D'] >= 0.15 and lo(t, 'all', 'changed') > 0 and g(t, 'all', 'shift')['D'] >= 0.03]
    S1 = [t for t in C if lo(t, 'all', 'changed') > 0]
    S2 = [t for t in C if g(t, 'all', 'changed')['D'] >= 0.05]
    S3 = [t for t in C if lo(t, 'L0-4', 'changed') > 0]
    S4 = [t for t in C if lo(t, 'all', 'shift') > 0]
    S5 = [t for t in C if O[t]['sets']['all']['changed_D_mass_adjusted_ci'][0] > 0]
    if len(S0) < 4: read = 'INSENSITIVE (S0 fails): nothing about C is read'
    elif len(S1) < 6: read = 'NO DETECTABLE stage-specific causal effect of the CFOP path on moves'
    elif len(S2) < 6: read = ('REAL BUT SMALL: a stage-specific causal effect exists but is below 0.05; '
                              + ('survives without the final layer' if len(S3) >= 6 else 'NOT shown without the final layer') + '; '
                              + ('representation moves toward the donor stage' if len(S4) >= 6 else 'representation NOT shown to move toward the donor stage'))
    else: read = 'SUBSTANTIAL stage-specific causal effect of the path'
    if len(S1) >= 6 and len(S0) >= 4:
        read += ('; survives adjustment for the router-mass gap (S5)' if len(S5) >= 6 else
                 '; NOT shown beyond the router-mass gap (S5 fails): accounted for by the recipient router preferring same-stage donor pairs')
    print('model | set: n d | changed cross/same D [CI] | KL cross/same | dlogp D | probe shift D [CI] | D half A/B')
    for t in K + C:
        o = O[t]; print(f"{t}  clean acc {o['clean_acc']:.3f} probe acc {o['probe_acc']:.3f}")
        for s, v in o['sets'].items():
            c = v['changed']; sh = v['shift']
            if s == 'all' and 'changed_D_mass_adjusted_ci' in v:
                print(f"   mass on donor pair cross/same {v['mass_on_donor_pair']['cross']:.3f}/{v['mass_on_donor_pair']['same']:.3f}; "
                      f"mass-adjusted D {v['changed_D_mass_adjusted']:+.3f} {v['changed_D_mass_adjusted_ci']}")
            print(f"   {s:5s}: {v['n_matched']:5d} {v['d_mean']:.2f} | {c['cross']:.3f}/{c['same']:.3f} {c['D']:+.3f} [{c['ci'][0]:+.3f} {c['ci'][1]:+.3f}] | "
                  f"{v['kl']['cross']:.3f}/{v['kl']['same']:.3f} | {v['dlogp']['D']:+.3f} | {sh['D']:+.3f} [{sh['ci'][0]:+.3f} {sh['ci'][1]:+.3f}] | "
                  f"{c['D_half']['A']:+.3f}/{c['D_half']['B']:+.3f}")
    v = dict(S0=f'{len(S0)}/5', S1=f'{len(S1)}/8', S2=f'{len(S2)}/8', S3=f'{len(S3)}/8', S4=f'{len(S4)}/8', S5=f'{len(S5)}/8', verdict=read)
    print(json.dumps(v, indent=1)); json.dump(v, open(f'{OUT}/report.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
