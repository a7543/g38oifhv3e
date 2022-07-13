#!/usr/bin/bash
ps -aux | grep colossalai | awk '{print $2}' | xargs kill
