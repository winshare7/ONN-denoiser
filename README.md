

# 项目介绍
    作者：汤文轩
    单位：清华大学精仪系
    邮箱：tangwx21@mails.tsinghua.edu.cn
    本项目是论文“All-optical image denoising using a diffractive visual processor”的复现，时间有限，仅大致实现了网络的基本结构和功能。
    论文链接：https://doi.org/10.1038/s41377-024-01385-6
    使用MINST手写数字数据集，添加椒盐噪声。预处理后的数据结构应为：112*112*60000；包含60000个112*112的灰度图。
    使用英伟达3090运行，建议使用矩池云租借GPU，方便快捷。
 
# 环境依赖
    pytorch 2.12
    python 3.8
# 目录结构描述
    ├── ReadMe.md           // 帮助文档
    
    ├── ONN denoiser 包含train_model.py和performance.py文件    // 用于模型训练
    
    ├── support image             // 包含部分去噪图片效果
    
    
 
