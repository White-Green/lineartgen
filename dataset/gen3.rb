require "bundler/inline"
require "erb"
require "json"
require "fileutils"
require "net/http"
require "uri"

gemfile do
    source "https://rubygems.org"
    gem "ruby-vips"
end

STAGES = {
    "2x" => {
        input_dir: "images",
        output_dir: "images_2x",
        model_name: "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
        scale: 2,
    },
    "4x" => {
        input_dir: "images",
        output_dir: "images_4x",
        model_name: "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        scale: 4,
    },
    "8x" => {
        input_dir: "images",
        output_dir: "images_8x",
        model_name: "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
        scale: 8,
    },
    "16x" => {
        input_dir: "images_2x",
        output_dir: "images_16x",
        model_name: "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
        scale: 8,
    },
    "32x" => {
        input_dir: "images_4x",
        output_dir: "images_32x",
        model_name: "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
        scale: 8,
    },
}.freeze

class ComfyUiClient
    def initialize
        @http = Net::HTTP.new(ENV.fetch("COMFYUI_API_HOST"), ENV.fetch("COMFYUI_API_PORT"))
        @http.open_timeout = 10
        @http.read_timeout = 60 * 60 * 24
        @http.write_timeout = 60 * 60 * 24
        @root_dir = ENV.fetch("COMFYUI_ROOT_DIR")
        @poll_interval = Float(ENV.fetch("COMFYUI_POLL_INTERVAL", "10"))
        raise "COMFYUI_POLL_INTERVAL must be positive" unless @poll_interval.positive?
    end

    def upload_image(image_path, remote_filename)
        File.open(image_path, "rb") do |file|
            request = Net::HTTP::Post.new("/upload/image")
            request.set_form(
                [
                    ["image", file, { filename: remote_filename, content_type: "image/png" }],
                    ["overwrite", "true"],
                    ["type", "input"],
                ],
                "multipart/form-data"
            )
            request_json(request)
        end
    end

    def enqueue(workflow)
        request = Net::HTTP::Post.new("/prompt", { "Content-Type" => "application/json" })
        request.body = workflow
        response = request_json(request)
        response.fetch("prompt_id")
    end

    def wait_for_output(prompt_id)
        loop do
            job = request_json(Net::HTTP::Get.new("/api/jobs/#{prompt_id}"))
            case job.fetch("status")
            when "pending", "in_progress"
                sleep(@poll_interval)
            when "completed"
                return job.fetch("preview_output")
            when "failed", "cancelled"
                message = job.dig("execution_error", "exception_message") || job.fetch("status")
                raise "ComfyUI job #{prompt_id} failed: #{message}"
            else
                raise "Unknown ComfyUI job status: #{job.fetch("status").inspect}"
            end
        end
    end

    def download_and_normalize(preview, destination, expected_width, expected_height)
        token = "#{Process.pid}-#{Thread.current.object_id}"
        downloaded = "#{destination}.download-#{token}.png"
        normalized = "#{destination}.normalized-#{token}.png"

        query = URI.encode_www_form(
            filename: preview.fetch("filename"),
            subfolder: preview.fetch("subfolder", ""),
            type: preview.fetch("type", "output")
        )
        request = Net::HTTP::Get.new("/view?#{query}")
        @http.request(request) do |response|
            ensure_success!(response)
            File.open(downloaded, "wb") do |file|
                response.read_body { |chunk| file.write(chunk) }
            end
        end

        image = Vips::Image.new_from_file(downloaded, access: :sequential)
        actual_size = [image.width, image.height]
        expected_size = [expected_width, expected_height]
        raise "Unexpected output size: #{actual_size.inspect}, expected #{expected_size.inspect}" unless actual_size == expected_size

        bands = image.bandsplit
        gray = if bands.length >= 3
            bands[0] * 0.2126 + bands[1] * 0.7152 + bands[2] * 0.0722
        else
            bands.fetch(0)
        end
        gray = gray.round(:rint).cast(:uchar)
        rgb = gray.bandjoin(gray).bandjoin(gray)
        rgb.pngsave(normalized, compression: 6, strip: true)
        File.rename(normalized, destination)
    ensure
        FileUtils.rm_f(downloaded) if downloaded
        FileUtils.rm_f(normalized) if normalized
    end

    def remove_uploaded_file(upload)
        remove_local_file("input", upload["subfolder"], upload["name"])
    end

    def remove_output_file(preview)
        return unless preview.fetch("type", "output") == "output"

        remove_local_file("output", preview["subfolder"], preview["filename"])
    end

    private

    def request_json(request)
        response = @http.request(request)
        ensure_success!(response)
        JSON.parse(response.body)
    end

    def ensure_success!(response)
        return if response.is_a?(Net::HTTPSuccess)

        raise "ComfyUI request failed: HTTP #{response.code} #{response.message}"
    end

    def remove_local_file(type, subfolder, filename)
        return if filename.nil? || filename.empty?

        root = File.expand_path(type, @root_dir)
        path = File.expand_path(File.join(subfolder || "", filename), root)
        return unless path.start_with?("#{root}#{File::SEPARATOR}")

        FileUtils.rm_f(path)
    end
end

def selected_ids(arguments)
    return arguments.map { |argument| Integer(argument, 10) }.uniq unless arguments.empty?

    File.readlines("#{__dir__}/gen/enabled_prompts.csv", chomp: true)
        .reject(&:empty?)
        .map { |line| Integer(line, 10) }
end

def image_size(path)
    image = Vips::Image.new_from_file(path, access: :sequential)
    [image.width, image.height]
end

def run_stage(client, workflow_template, stage_name, stage, ids, run_id)
    input_dir = "#{__dir__}/#{stage.fetch(:input_dir)}"
    output_dir = "#{__dir__}/#{stage.fetch(:output_dir)}"
    FileUtils.mkdir_p(output_dir)
    jobs = []
    failures = []

    begin
        ids.each do |id|
            basename = format("%04d.png", id)
            input_path = "#{input_dir}/#{basename}"
            output_path = "#{output_dir}/#{basename}"
            raise "Input image not found for #{stage_name}: #{input_path}" unless File.file?(input_path)

            width, height = image_size(input_path)
            expected_width = width * stage.fetch(:scale)
            expected_height = height * stage.fetch(:scale)

            if File.file?(output_path)
                actual_size = image_size(output_path)
                expected_size = [expected_width, expected_height]
                raise "Existing output has unexpected size: #{output_path} #{actual_size.inspect}, expected #{expected_size.inspect}" unless actual_size == expected_size

                puts "Skipping #{stage_name} #{basename}"
                next
            end

            puts "Queueing #{stage_name} #{basename}"
            STDOUT.flush

            remote_filename = "lineartgen-gen3-#{run_id}-#{stage_name}-#{basename}"
            upload = client.upload_image(input_path, remote_filename)
            begin
                workflow = workflow_template.result_with_hash(
                    model_name: stage.fetch(:model_name),
                    image_filename: upload.fetch("name"),
                    filename: "lineartgen-gen3/#{run_id}/#{stage_name}-#{format("%04d", id)}",
                )
                jobs << {
                    basename:,
                    expected_height:,
                    expected_width:,
                    output_path:,
                    prompt_id: client.enqueue(workflow),
                    upload:,
                }
            rescue StandardError
                client.remove_uploaded_file(upload)
                raise
            end
        end
    rescue StandardError => error
        warn "Stopped queueing #{stage_name}: #{error.message}"
        failures << error
    end

    puts "Queued #{jobs.length} #{stage_name} job(s)"
    STDOUT.flush

    jobs.each do |job|
        preview = nil
        begin
            puts "Waiting for #{stage_name} #{job.fetch(:basename)}"
            STDOUT.flush
            preview = client.wait_for_output(job.fetch(:prompt_id))

            puts "Post-processing #{stage_name} #{job.fetch(:basename)}"
            STDOUT.flush
            client.download_and_normalize(
                preview,
                job.fetch(:output_path),
                job.fetch(:expected_width),
                job.fetch(:expected_height)
            )
        rescue StandardError => error
            warn "Failed #{stage_name} #{job.fetch(:basename)}: #{error.message}"
            failures << error
        ensure
            client.remove_output_file(preview) if preview
            client.remove_uploaded_file(job.fetch(:upload))
        end
    end

    raise failures.first unless failures.empty?
end

stage_argument = ARGV.shift || "all"
stage_names = stage_argument == "all" ? STAGES.keys : [stage_argument]
unknown_stages = stage_names - STAGES.keys
raise "Unknown stage #{unknown_stages.join(", ")}; expected one of: all, #{STAGES.keys.join(", ")}" unless unknown_stages.empty?

ids = selected_ids(ARGV)
workflow_template = ERB.new(File.read("#{__dir__}/gen/workflow3.json.erb"))
client = ComfyUiClient.new
run_id = "#{Time.now.strftime("%Y%m%d%H%M%S")}-#{Process.pid}"

stage_names.each do |stage_name|
    run_stage(client, workflow_template, stage_name, STAGES.fetch(stage_name), ids, run_id)
end
