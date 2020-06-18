#!/bin/bash

for s in {1..10} ; do
  python3 gen_frame.py colour $(( 160 * $s )) $(( 90 * $s ))
done