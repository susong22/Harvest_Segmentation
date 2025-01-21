class HarvestBase:
    SPLITS = ['train', 'test']
    
    # Number of classes without background
    N_CLASSES = 3
    
    # Directory structure
    RGB_DIR    = 'rgb'
    DEPTH_DIR  = 'depth' 
    LABELS_DIR = 'labels'
    
    # Pallette
    CLASS_NAMES = [
        'void',   # 0
        'field',  # 1
        'machine' #2
    ]
    
    CLASS_COLORS = [
        [0, 0, 0],      # void - black
        [0, 255, 0],    # field - green
        [0, 0, 255]     # machine - blue
    ]