# RetinaNet
Tensorflow implement of RetinaNet
usage
details about code structure and how to use in usage.doc
pretrained weight
I use tensorflow's official weight, You can choose resnet weight or coco trained RetinaNet. Donwload them. 
sorport
1.multigpu training
2.data augmentation(random crop/flip/rotate/color augment...)
spped
13fps/s in RTX2060.
performance
eval map:~0.73 in voc07 (test) and trained in voc12(train+val)+voc07(train + val) with a batch size of only 4 and with data augment of only left-right flip. More augment and larger batchsize might get a much better result.You can follow the usage direction to get the same result
Results

referrence code
1.Tensorflow object API
2.Latter, I will put on them...
