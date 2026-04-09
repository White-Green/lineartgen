require "csv"
require "set"

seed_column_number = Integer(ARGV.fetch(0))
if seed_column_number < 1
    raise ArgumentError, "seed column number must be >= 1"
end

# CSV index:
#   0 => prompt id
#   1 => seed1
#   2 => seed2
#   ...
target_column_index = seed_column_number

recalc_target = File.readlines("#{__dir__}/gen/enabled_prompts.csv").map(&:strip).map(&:to_i).to_set
csv = CSV.generate do |csv|
    CSV.foreach("#{__dir__}/gen/prompts.csv") do |row|
        row = row.to_a
        id = row[0]
        if recalc_target.include?(id.to_i)
            row[target_column_index] = ((rand * 900_000_000_000_000).to_i + 100_000_000_000_000).to_s
        end
        csv << row
    end
end

File.write("#{__dir__}/gen/prompts.csv", csv)
