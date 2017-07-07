#!/bin/bash
for file in ~/Programs/ChangHui_NCRCRG/ChangHui_files/*
do
if [ $i =='*.nd2' ];then
  ./bfconvert -nolookup "$file" "$file".tif 
fi
done

for file in ~/Programs/ChangHui_NCRCRG/ChangHui_files/*
do
if [ $i =='*.tif' ];then
  ./analyze  "$file"  
fi
done

