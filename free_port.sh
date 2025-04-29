#!/bin/bash
# Tìm và dừng các tiến trình trên cổng 5000
lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs -I {} kill -9 {}