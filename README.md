# Universal_Adversarial_Perturbations-Pytorch
Implement of "Universal adversarial perturbations" on CIAFR10 in Pytorch.

## TODO：
### Generation_via_optimization.py  
Nothing.This idea totally failed.  
It's amazing that the found universal perturbation even performs worse than the random perturbation while attacking untargetedly.  
And while attacking targetedly, the successful attacking rate is about 10.00%, which is the percentage of the target label pictures in the dataset, which also means failure.
### Generation.py
Nothing. Failed.  
While following paper[1] to attack with DeepFool, I don't know why it cannot converge on the training set.  
And while attacking with FGSM, a well-performed universal adversarial perturbation is found on the training set.  
But I don't know why the found perturbation cannot generalize to the test set.  
I even doubt that the algorithm cannot work on CIFAR10 at all.
## Reference:
[1] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, Pascal Frossard  
      [Universal adversarial perturbations. arXiv:1610.08401](https://arxiv.org/abs/1610.08401)  
[2] Jonas Rauber, Wieland Brendel, Matthias Bethge  
      [Foolbox: A Python toolbox to benchmark the robustness of machine learning models. arXiv:1707.04131](https://arxiv.org/abs/1707.04131)
