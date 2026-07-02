"""
VLM-based VideoQA evaluation for TASKER.

Runs the TASKER keyframe search (and Uniform / VideoTree / VideoAgent /
Text-only baselines) with a multi-image vision-language model on the
EgoSchema and NExT-QA benchmarks.

Methods:
  - textonly   : blind baseline, no frames (lower bound)
  - uniform    : uniformly sampled frames + VLM
  - videotree  : adaptive CLIP-clustering frame selection + VLM
  - videoagent : VLM-guided iterative frame selection + VLM
  - tasker     : TASKER keyframe search (A*) + VLM (ours)
"""

__version__ = "1.0.0"
