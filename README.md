# Game of Movement

Uses the movement detected in a video by considering the difference between subsequent frames to trigger n evolution cycles of game of life as implemented in [https://github.com/ctjacobs/game-of-life](https://github.com/ctjacobs/game-of-life)

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
