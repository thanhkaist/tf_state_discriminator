State discriminator
=============================
Framework: tensorflow \
This code provide training and testing code to create 2 classifier as below: 

### Target discriminator  
Input: state image, target image \
Output: 2 classes \
- 0: state image is not target state 
- 1: state image is target state 

### Neighbor discriminator 
Input: state image 1, state image 2 \
Output:2 classes \
0: 2 state images are not neighbor \
1: 2 state images are neighbor 

To run the code. Please checkout main.py


 
