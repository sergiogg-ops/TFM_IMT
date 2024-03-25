from sb_mbart import check_segments

target = 'Hoy esta nublado pero estoy feliz'
hyp = 'Como hoy esta nublado estoy triste'

print(check_segments(target,hyp))