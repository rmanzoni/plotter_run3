for e in $(conda env list | awk '!/^#/ && NF {print $NF}'); do
  printf '%-55s ' "$e"
  "$e/bin/python" -c "import uproot, mplhep, matplotlib, numpy; print('OK', uproot.__version__)" 2>/dev/null || echo "missing/no python"
done
