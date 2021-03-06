## <b>Computer Vision - Soccer Analytics</b> [[Project Page]](https://github.com/cchadd/tracking) <br>

Clément Chadebec, Thibault Desfontaines

![Teaser Image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTThjic_ml5y4jsOK149EIkaqkQT5NfzonORcszj5V_g0hgb8ny)

### Overview ###
This repository contains:

<b>Calibration for an indoor soccer-field
 - (0) Camera calibration script using OpenCV
 - (1) Interest points selection

<b>Interaction with Yolo detection

<b>Detection of the coordinates of the players

<b>Appendices</b>
 - (A) Related follow-up work
 
### Training videos ###
From [Urban Soccer](https://www.urbansoccer.fr/videos/)
<ul>
    <li>  Aubervilliers, Terrain 10, 03/05/2019, 20h13</li>
</ul>

### Dependencies ###
This code makes use of the Yolo detector available at [pjreddie webpage](https://pjreddie.com/darknet/) as well as OpenCV for calibration and to apply filters on the image.

## Calibration ##

### (0) Run a calibration ###
To start calibration, run 'python hello.py'.

### (1) Evaluate the result on a Notebook ###
Open the following notebook 

### (2) Choose a ROI on a video ###

### Draw a perspective grid on the field ###
https://web.archive.org/web/20160418004152/http://freespace.virgin.net/hugo.elias/graphics/x_persp.htm



## Related follow-up work ##

### References ###

## Code structure ##

<ul>
  <li>execution path
  <ul> 
    <li> darknet </li>
    <li> roi_frames </li>
    <li> videos </li>
    <li> frames </li>
    <li> shared/tracking 
      <ul> 
        <li> .git </li>
        <li> calibration </li>
        <li> ball_detection </li>
        <li> player_detection </li>
        <li> player_stats </li>
        <li> gui </li>
        <li> experiments </li>
        <li> references </li>
        <li> AR </li>        
      </ul>
    </li>
  </ul>
  </li>
</ul> 

