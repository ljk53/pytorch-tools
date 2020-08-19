#!/bin/bash

set -eu -o pipefail

ROOT="$( cd "$(dirname "$0")"; pwd -P)"

cd $ROOT

cat microbench.md | awk -f <(cat - <<-'EOF'
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

function compare(avg, _keyword, _replace, comment, reverse) {
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
        if (reverse) {
          tmp = k; k = other; other = tmp;
        }
        printf "%-30s\t%40s\t%10.2f ns\t%10.0f%%\n", comment, other, avg[other] - avg[k], (avg[other] - avg[k]) / avg[k] * 100
      }
    }
  }
}

END {
  for (k in count) {
    avg[k] = duration[k] / count[k]
  }
  compare(avg, "_novar", "", "Autograd Dispatching Cost", 0)
  compare(avg, "_outplace", "", "Output Allocation Cost", 0)
  compare(avg, "_nograd", "_grad", "Autograd Cost", 0)
  compare(avg, "_scripted", "", "Python v.s. TorchScript", 0)
  compare(avg, "cpp", "py", "Python v.s. CPP", 0)
  compare(avg, "py,_scripted", "cpp,", "TorchScript v.s. CPP", 1)
}
EOF) | sort -t$'\t' -k3 -n | sort -t$'\t' -k1,1 -s
