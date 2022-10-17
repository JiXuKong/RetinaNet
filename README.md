# RetinaNet
Tensorflow implement of RetinaNet<br>
## usage
details about code structure and how to use in usage.doc<br>
## pretrained weight
I use tensorflow's official weight, You can choose resnet weight or coco trained RetinaNet. Donwload them. <br>
## sorport
1.multigpu training<br>
2.data augmentation(random crop/flip/rotate/color augment...)<br>
## spped
13fps/s in RTX2060.<br>
## performance
eval map(pretrained resnet50):~0.73 in voc07 (test) and trained in voc12(train+val)+voc07(train + val)<br>
eval map(coco pretrained):~0.81 in voc07 (test) and trained in voc12(train+val)+voc07(train + val)<br>
Batch size:4<br>
data augmentation: flip<br>
## Results

## referrence code
1.Tensorflow object API<br>
2.Latter, I will put on them...<br>
