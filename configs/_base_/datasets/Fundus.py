Fundus = ['Drishti_GS', 'ORIGA', 'REFUGE', 'REFUGE_Valid', 'RIM_ONE_r3']

dataset = dict(
    type='Drishti_GS',
    root='./data/Segmentation/MedicalSegmentation/Fundus',
    image_size=512,  # 目前看就是谨慎选择img_size，Swin TRM只能接受384的输入
)

# clip_prompts = './clip_prompts/funds.json'

