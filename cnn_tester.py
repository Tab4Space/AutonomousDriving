import numpy as np
from helper import Agent
from keras.models import load_model

direction_dict = {
    '0':'straight',
    '1':'right',
    '2':'left',
    '3':'left change',
    '4':'right change'
}

def main():
    agent = Agent(useAPI=False)
    model = load_model('./models/direction/class2.h5')

    while True:
        img = agent.getCamImage({'F_myCamSeg':5})[:, :, :, :3]
        pred = model.predict([img])
        pred = np.argmax(pred[0])

        print(direction_dict[str(pred)])

if __name__ == '__main__':
    main()