import sys
from art import aggregators
from art import scores
from art import significance_tests

if len(sys.argv) < 3 or len(sys.argv) > 4:
    sys.stderr.write('Usage: ' + sys.argv[0]
                     + 'system_1 system_2 {repetitions}.\n')
    sys.stderr.write(' repetitions: total trials to compute (default: 10000.)\n')
    sys.exit(-1)

if len(sys.argv) == 4:
    trials = int(sys.argv[3])
else:
    trials = 10000

test = significance_tests.ApproximateRandomizationTest(
    scores.Scores.from_file(open(sys.argv[1])),
    scores.Scores.from_file(open(sys.argv[2])),
    aggregators.average,
    trials=trials)

print("\t Significance level:",  test.run())
