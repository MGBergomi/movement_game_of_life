# Game of Movement

[![Audi R8](http://img.youtube.com/vi/FxOfgN-yevk/0.jpg)](https://www.youtube.com/watch?v=KOxbO0EI4MA "Audi R8")

Uses the movement detected in a video by considering the difference between subsequent frames to trigger n evolution cycles of game of life as implemented in [https://github.com/ctjacobs/game-of-life](https://github.com/ctjacobs/game-of-life).

Uses blockwise_view as implemented in [https://github.com/ilastik/lazyflow/](https://github.com/ilastik/lazyflow/blob/master/lazyflow/utility/blockwise_view.py)

###### Remark:
The current GameOfLife class has a series of additional methods wrt to the original implementation, but no substantial changes


### Install

- clone this repository
- In your terminal type
  ```
  pip install -r requirements.txt
  ```

### Usage

##### GOL on webcam stream

```
p = Preprocess(path = None, cols = 20, rows = 10)
p.process_video()
```

##### GOL on webcam video

```
p = Preprocess(path = 'path_to_video', cols = 20, rows = 10)
p.process_video()
```

##### Visualisation

Visualisation is done with opencv

