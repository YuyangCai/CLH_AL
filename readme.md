

--cpt False 不加载模型
--save false 不保存模型

VGG-FACE2 初始每个文件夹选一张，后续每个文件夹选50张 blur Entropy 方法 8 个cycle 
nohup python /home/cyy/A_studio/code/AL_VGG-Face2/main_resume.py --uncertainty Entropy --dataset VGGDataset --subnum 1  --query 50 --epoch 40 --cycle 8 --device cuda:2 > vgg_blur_en_50_2.log 2>&1 &




不保存权重，测试entropy class_en class_score 的效果

nohup python /home/cyy/A_studio/code/AL_VGG-Face2/main_resume.py --uncertainty Entropy  --dataset VGGDataset --subnum 1  --query 1 --epoch 100 --cycle 16 --device cuda:3 --cpt False --save False > vgg_en_nocpt.log 2>&1 &

nohup python /home/cyy/A_studio/code/AL_VGG-Face2/main_resume.py --uncertainty Class_Entropy  --dataset VGGDataset --subnum 1  --query 1 --epoch 100 --cycle 16 --device cuda:3 --cpt False --save False > vgg_classen_nocpt.log 2>&1 &

nohup python /home/cyy/A_studio/code/AL_VGG-Face2/main_resume.py --uncertainty class_score  --dataset VGGDataset --subnum 1  --query 1 --epoch 100 --cycle 16 --device cuda:3 --cpt False --save False  > vgg_class_score_nocpt.log 2>&1 &


nohup python /home/cyy/A_studio/code/AL_VGG-Face2/main_resume.py --uncertainty Entropy  --dataset Cifar10  --subnum 1000  --query 1000 --epoch 200 --cycle 40 --device cuda:1 --cpt False --save False  --nw 4  > cifar10_en.log 2>&1 &

nohup python /home/cyy/A_studio/code/AL_VGG-Face2/main_resume.py --uncertainty Margin  --dataset Cifar10  --subnum 1000  --query 1000 --epoch 200 --cycle 40 --device cuda:1 --cpt False --save False  --nw 4  > cifar10_ma.log 2>&1 &

nohup python /home/cyy/A_studio/code/AL_VGG-Face2/main_resume.py --uncertainty class_margin  --dataset Cifar10  --subnum 1000  --query 1000 --epoch 200 --cycle 40 --device cuda:1 --cpt False --save False  --nw 4  > cifar10_class_ma.log 2>&1 &