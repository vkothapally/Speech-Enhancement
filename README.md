Single channel speech dereverberation
========

In this study, we propose such a preprocessing technique for combined detection and enhancement of reverberant speech using a single microphone for distant speech applications. The proposed system employs a framework where the target speech is synthesized using continuous auditory masks estimated from sub-band signals. 

Linear gammatone analysis/synthesis filter banks are used as an auditory model for sub-band processing. Features for each of these sub- band signals are extracted from short-time frames which are shifted in time. An apt consolidation of these extracted features are used in estimating the probability of a frame comprising source’s direct path, which is later used as a continuous auditory mask to preserve & retrieve the direct path from the reverberant signal. The speech activity is determined by clustering frames with high probability. The sub-band signals with an auditory mask applied are passed through a set of synthesis filters to reconstruct enhanced speech signal.


Installation
========
For testing and/or making changes to the most recent version: Clone the repository and install it as follows:

```
  git clone https://github.com/vkothapally/Speech-Dereverberation.git
  cd Speech-Dereverberation
  pip install --editable .
```

Usage
========


Results
========
<audio src="./audiofiles/Clean.wav" controls preload>ClickThis</audio>

Citation
========

If this implementation was helpful in your research, you can cite the following paper:

     @article{kothapally2017speech,
      title={Speech detection and enhancement using single microphone for distant speech applications in reverberant environments},
      author={Kothapally, Vinay and Hansen, John HL},
      journal={Proc. Interspeech 2017},
      pages={1948--1952},
      year={2017}
    }


