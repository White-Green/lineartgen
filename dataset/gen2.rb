require "bundler/inline"
require "erb"
require "csv"
require "net/http"
require "uri"
require "json"
require "fileutils"

gemfile do
    source "https://rubygems.org"
    gem "ruby-vips"
end

def upload_image_to_comfyui(http, image_path)
    file = File.open(image_path, "rb")
    req = Net::HTTP::Post.new("/upload/image")
    req.set_form(
        [
            ["image", file, { filename: File.basename(image_path), content_type: "image/png" }],
            ["overwrite", "true"],
        ],
        "multipart/form-data"
    )
    res = http.request(req)
    JSON.parse(res.body)
ensure
    file&.close
end

workflow_template = ERB.new(File.read("#{__dir__}/gen/workflow2.json.erb"))
comfyui = Net::HTTP.new(ENV.fetch("COMFYUI_API_HOST"), ENV.fetch("COMFYUI_API_PORT"))

input_dir = "#{__dir__}/images_colored"
output_dir = "#{__dir__}/images"
FileUtils.mkdir_p(output_dir)

prompt_id = []
filename_prefix = Time.now.strftime("%Y%m%d%H%M%S")
prompts = CSV.foreach("#{__dir__}/gen/prompts.csv").map { |i, _, seed| [i.to_i, seed] }.to_h

File.readlines("#{__dir__}/gen/enabled_prompts.csv").map(&:strip).map(&:to_i).each do |i|
    seed = prompts[i]
    image_path = "#{input_dir}/#{format("%04d", i)}.png"
    raise "Input image not found: #{image_path}" unless File.exist?(image_path)

    uploaded = upload_image_to_comfyui(comfyui, image_path)

    workflow = workflow_template.result_with_hash(
        seed:,
        image_filename: uploaded["name"],
        filename: "#{filename_prefix}_#{format("%04d", i)}",
    )
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
        image = Vips::Image.new_from_buffer(file.body, "")
        r, g, b = image.bandsplit[0,3]
        r = r.cast(:float)
        g = g.cast(:float)
        b = b.cast(:float)
        minc = (r < g).ifthenelse(r, g)
        minc = (b < minc).ifthenelse(b, minc)
        maxc = (r > g).ifthenelse(r, g)
        maxc = (b > maxc).ifthenelse(b, maxc)
        
        use_minc = minc < -(maxc - 255)
        gray = use_minc.ifthenelse(minc, maxc).cast(image.format)
        
        rgb_image = gray.bandjoin(gray).bandjoin(gray)
        rgb_image.write_to_file("#{__dir__}/images/#{format("%04d", i)}.png")
        break
    end
end
