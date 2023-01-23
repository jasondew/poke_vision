defmodule PokeVision do
  @moduledoc """
  Documentation for `PokeVision`.
  """

  require Axon

  # RGBa
  @width 734
  @height 1024
  @channels 3

  def load_data do
    File.stream!("train.csv")
    |> NimbleCSV.RFC4180.parse_stream()
    |> Enum.map(fn [id, image_url, _caption, name, _hp, _set_name] ->
      %{id: id, image_url: image_url, name: name}
    end)
  end

  def get_sample(size) do
    load_data()
    |> Enum.shuffle()
    |> Enum.take(size)
  end

  def labels(data) do
    data |> Enum.map(& &1.name) |> Enum.uniq()
  end

  def tensorize(data, labels, batch_size) do
    data
    |> Stream.chunk_every(batch_size, batch_size, :discard)
    |> Task.async_stream(
      fn batch ->
        {input, labels} =
          batch
          |> Enum.map(fn pokemon -> tensorize_pokemon(pokemon, labels) end)
          |> Enum.unzip()

        {Nx.stack(input), Nx.stack(labels)}
      end,
      max_concurrency: 3
    )
    |> Stream.map(fn {:ok, {input, labels}} -> {input, labels} end)
    |> Stream.cycle()
  end

  defp tensorize_pokemon(pokemon, labels) do
    {:ok, request} = Finch.build(:get, pokemon.image_url) |> Finch.request(PokeVision.Finch)
    {:ok, image} = Image.open(request.body)

    {:ok, resized_image} = Image.thumbnail(image, "#{@width}x#{@height}", resize: :force)

    {image_without_alpha, _alpha} = Image.split_alpha(resized_image)
    {:ok, final_image} = Image.to_colorspace(image_without_alpha, :srgb)

    {:ok, tensor} = Image.to_nx(final_image, shape: :hwc)

    class =
      labels
      |> Enum.map(&if &1 == pokemon.name, do: 1, else: 0)
      |> Nx.tensor(type: {:u, 8})

    {tensor, class}
  end

  defp build_model(input_shape, label_count) do
    Axon.input("input", shape: input_shape)
    |> Axon.conv(16, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    |> Axon.spatial_dropout(rate: 0.5)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.spatial_dropout(rate: 0.5)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.flatten()
    |> Axon.dropout(rate: 0.5)
    |> Axon.dense(512, activation: :relu)
    |> Axon.dense(label_count, activation: :softmax)
  end

  defp train_model(model, data, optimizer, epochs, iterations_per_epoch) do
    model
    |> Axon.Loop.trainer(:binary_cross_entropy, optimizer, :identity, log: 1)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: iterations_per_epoch)
  end

  def test_model(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), model_state)
  end

  def run(sample_size \\ 100, batch_size \\ 10, epochs \\ 10, iterations_per_epoch \\ 10) do
    data = get_sample(sample_size)
    all_labels = labels(data)
    tensorized_data = tensorize(data, all_labels, batch_size)

    model = build_model({nil, @height, @width, @channels}, Enum.count(all_labels)) |> IO.inspect()
    template = tensorized_data |> Enum.take(1) |> List.first() |> elem(0) |> Nx.to_template()

    IO.puts(Axon.Display.as_table(model, template))

    optimizer = Axon.Optimizers.adam(1.0e-4)
    centralized_optimizer = Axon.Updates.compose(Axon.Updates.centralize(), optimizer)

    model_state =
      train_model(model, tensorized_data, centralized_optimizer, epochs, iterations_per_epoch)
      |> IO.inspect()

    {test_images, test_labels} =
      data
      |> tensorize(all_labels, batch_size)
      |> Enum.take(batch_size)
      |> Enum.unzip()

    test_model(model, model_state, test_images, test_labels)
  end
end
