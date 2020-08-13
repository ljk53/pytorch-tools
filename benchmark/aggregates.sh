#!/bin/bash

set -eu -o pipefail

cat README.md | awk -f <(cat - <<-'EOF'
/###/ {
  cpu = $2
}

/C++|Python/ {
  if ($0 ~ /Python/) {
    lang = "py"
  } else {
    lang = "cpp"
  }
  if ($0 ~ /NUM_THREADS/) {
    thread = "1"
  } else {
    thread = "n"
  }
}

/[a-z_]+ +[0-9.]+ +[0-9.]+/ {
  # key = cpu ":" thread ":" lang ":" $1
  key = cpu ":" lang ":" $1
  count[key] += 1
  duration[key] += $3
}

function compare(avg, _keyword, _replace, comment) {
  split(_keyword, keywords, ",")
  split(_replace, replaces, ",")

  for (k in avg) {
    other = k
    matched = 1
    for (i in keywords) {
      if (other !~ keywords[i]) {
        matched = 0
        break
      }
      gsub(keywords[i], replaces[i], other)
    }
    if (matched) {
      if (other in avg) {
        printf "%-30s\t%40s\t%10.2f ns\t%10.0f%%\n", comment, other, avg[other] - avg[k], (avg[other] - avg[k]) / avg[k] * 100
      }
    }
  }
}

END {
  for (k in count) {
    avg[k] = duration[k] / count[k]
  }
  compare(avg, "_novar", "", "Autograd Dispatching Cost")
  compare(avg, "_outplace", "", "Output Allocation Cost")
  compare(avg, "_nograd", "_grad", "Autograd Cost")
  compare(avg, "_scripted", "", "TorchScript v.s. Python")
  compare(avg, "cpp", "py", "CPP v.s. Python")
  compare(avg, "py,_scripted", "cpp,", "TorchScript v.s. CPP")
}
EOF) | sort -t$'\t' -k3 -n | sort -t$'\t' -k1,1 -s
