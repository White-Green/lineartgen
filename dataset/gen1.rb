require "erb"
require "csv"
require "net/http"
require "uri"
require "json"
require "fileutils"

workflow_template = ERB.new(File.read("#{__dir__}/gen/workflow1.json.erb"))
comfyui = Net::HTTP.new(ENV.fetch("COMFYUI_API_HOST"), ENV.fetch("COMFYUI_API_PORT"))

output_dir = "#{__dir__}/images_colored"
FileUtils.mkdir_p(output_dir)

prompt_id = []
filename_prefix = Time.now.strftime("%Y%m%d%H%M%S")

prompts = CSV.foreach("#{__dir__}/gen/prompts.csv").map { |i, seed, _, prompt| [i.to_i, [seed, prompt]] }.to_h
File.readlines("#{__dir__}/gen/enabled_prompts.csv").map(&:strip).map(&:to_i).each do |i|
    seed, prompt = prompts[i]
    prompt = prompt.strip
    workflow = workflow_template.result_with_hash(seed:, prompt:, filename: "#{filename_prefix}_#{format("%04d", i)}")
    response = comfyui.post("/prompt", workflow, { "Content-Type": "application/json" })
    response = JSON.parse(response.body)
    prompt_id << [i, response["prompt_id"]]
end

puts "All prompts are enqueued"
STDOUT.flush

prompt_id.each do |i, id|
    loop do
        job = comfyui.get("/api/jobs/#{id}")
        job = JSON.parse(job.body)
        case job["status"]
        when "in_progress"
            sleep(10)
            next
        when "failed"
            pp job
            break
        end

        puts "Downloading image #{format("%04d", i)}"
        STDOUT.flush
        filename = job["preview_output"]["filename"]
        file = comfyui.get("/view?filename=#{filename}")
        File.binwrite("#{output_dir}/#{format("%04d", i)}.png", file.body)
        break
    end
end
