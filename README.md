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
* New eval results train with multi-augment stratages and batch_size=4(voc 07 + 12 trainval and test in voc07 test set)  
mAP    | aero | bike | bird | boat | bottle | bus | car | cat | chir | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv   
:-----  |----: |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- |:---- :  
0.773    | 0.88 | 0.83 | 0.79 | 0.64 | 0.49 | 0.80 | 0.84 | 0.92 | 0.55 | 0.80 | 0.62 | 0.90 | 0.84 | 0.83 | 0.81 | 0.49 | 0.75 | 0.72 | 0.88 | 0.78   
## Results

## referrence code
1.Tensorflow object API<br>
2.Latter, I will put on them...<br>
