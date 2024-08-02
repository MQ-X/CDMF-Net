# CDMF-Net
Joint optic disc and optic cup segmentation based on boundary prior and adversarial learning.

## Purpose 
The most direct means of glaucoma screening is to use cup-to-disc ratio (CDR) via color fundus photography, the first step of which is the precise segmentation of the optic cup (OC) and optic disc (OD). In recent years, convolution neural networks (CNN) have shown outstanding performance in medical segmentation tasks. However, most CNN-based methods ignore the effect of boundary ambiguity on performance, which leads to low generalization. This paper is dedicated to solving this issue.

## Methods 
In this paper, we propose a novel segmentation architecture, called CDMF-Net, which introduces an auxiliary boundary branch and adversarial learning to jointly segment OD and OC in a multi-label manner. To generate more accurate results, the generative adversarial network (GAN) is exploited to encourage boundary and mask predictions to be similar to the ground truth ones.

## Results 
Experimental results show that our CDMF-Net system achieves stateof-the-art OC and OD segmentation performance on three publicly available datasets, i.e., the Dice scores for the optic disc/cup on the Drishti-GS, RIMONE-r3 and REFUGE(only train part) datasets are 0.9649/0.8897, 0.9579/0.8630, and 0.9617/0.8823, respectively.

## Conclusion
In this work, we not only achieve superior OD and OC segmentation results, but also confirm that the values calculated through the geometric relationship between the former two are highly related to glaucoma.
