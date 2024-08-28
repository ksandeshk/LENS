Running Experiments:

For LENS related options :

Flower/eval_with_lens_attribution_attack.py : choose ig_attack or ig_attack_with_lens while importing SaliencyAttack for top-k attack without / with LENS, respectively.

Flower/ig_attack_with_lens.py : to enable k-LENS while constructing the top-k attack uncomment the corresponding "submat" code in create_attack_ops.
