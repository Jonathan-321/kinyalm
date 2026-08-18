const DEFAULT_ROOT =
  "/Users/jonathanmuhire/Documents/RL/KinyaLM-data-recovery/outputs/generation/" +
  "gemini-web-longform-1m-v1";

const sleep = (tab, milliseconds) =>
  tab.playwright.waitForTimeout(milliseconds);

const readJson = async (fs, path) => {
  try {
    return JSON.parse(await fs.readFile(path, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
};

export async function createGeminiWebWorker(tab, options = {}) {
  const fs = await import("node:fs/promises");
  const root = options.root ?? DEFAULT_ROOT;
  const jobsPath = `${root}/web-jobs.jsonl`;
  const rawDir = `${root}/raw-web-jobs`;
  const rawGroupDir = `${root}/raw-web-groups`;
  const validatedDir = `${root}/validated-web-jobs`;
  const activePath = `${root}/active-web-job.json`;
  const acceptedModePattern = options.acceptedModePattern ?? /Pro/;
  const desiredMenuItemPattern = options.desiredMenuItemPattern ?? /3\.1 Pro/;
  const jobs = (await fs.readFile(jobsPath, "utf8"))
    .split("\n")
    .filter(Boolean)
    .map(JSON.parse);
  const byId = new Map(jobs.map((job) => [job.web_job_id, job]));

  await fs.mkdir(rawDir, { recursive: true });
  await fs.mkdir(rawGroupDir, { recursive: true });
  await fs.mkdir(validatedDir, { recursive: true });

  const writeState = (value) =>
    fs.writeFile(activePath, `${JSON.stringify(value, null, 2)}\n`);

  const verifyConversationIds = (text, job, label) => {
    const missing = job.expected_conversations
      .map((row) => row.conversation_id)
      .filter((conversationId) => !text.includes(conversationId));
    if (missing.length) {
      throw new Error(`${label} verification failed for ${job.web_job_id}`);
    }
  };

  const submit = async (job) => {
    const baseUrl = "https://gemini.google.com/app";
    await tab.goto(baseUrl);
    await sleep(tab, 1600);
    const modePicker = tab.playwright.getByRole("button", {
      name: /Open mode picker/,
    });
    let modeLabel = await modePicker.innerText({ timeoutMs: 15000 });
    if (!acceptedModePattern.test(modeLabel)) {
      const desiredMode = tab.playwright.getByRole("menuitem", {
        name: desiredMenuItemPattern,
      });
      if ((await desiredMode.count()) === 0) {
        await modePicker.click();
        await sleep(tab, 400);
      }
      if (await desiredMode.count()) {
        try {
          await desiredMode.click();
        } catch {
          throw new Error(`teacher mode is unavailable: ${modeLabel}`);
        }
        await sleep(tab, 600);
        modeLabel = await modePicker.innerText({ timeoutMs: 15000 });
      }
    }
    if (!acceptedModePattern.test(modeLabel)) {
      throw new Error(`teacher mode is not accepted: ${modeLabel}`);
    }
    const box = tab.playwright.getByRole("textbox", {
      name: "Enter a prompt for Gemini",
    });
    await box.waitFor({ state: "visible", timeoutMs: 20000 });
    await box.fill(job.prompt, { timeoutMs: 15000 });
    let draft = await box.innerText({ timeoutMs: 15000 });
    try {
      verifyConversationIds(draft, job, "draft");
    } catch {
      await tab.clipboard.writeText(job.prompt);
      await box.press("Meta+A");
      await box.press("Backspace");
      await box.press("Meta+V");
      await sleep(tab, 600);
      draft = await box.innerText({ timeoutMs: 15000 });
      verifyConversationIds(draft, job, "pasted draft");
    }
    await sleep(tab, 2500);
    await tab.playwright.getByRole("button", {
      name: "Send message",
    }).click();
    await sleep(tab, 2600);
    const sent = await tab.playwright.locator("user-query").last().innerText();
    verifyConversationIds(sent, job, "sent prompt");
    await writeState({
      status: "active",
      web_job_id: job.web_job_id,
      teacher_mode: modeLabel,
      submitted_at: new Date().toISOString(),
    });
  };

  const readLatestResponse = async () => {
    const code = tab.playwright.locator("message-content code");
    const codeCount = await code.count();
    const messages = tab.playwright.locator("message-content");
    const messageCount = await messages.count();
    return codeCount === 1
      ? code.nth(0).innerText()
      : messageCount
        ? messages.nth(messageCount - 1).innerText()
        : "";
  };

  const nextAttemptPath = async (directory, stem) => {
    const names = await fs.readdir(directory);
    const pattern = new RegExp(`^${stem}\\.attempt-(\\d+)\\.raw\\.txt$`);
    const attempt = Math.max(
      0,
      ...names.map((name) => Number(name.match(pattern)?.[1] ?? 0)),
    ) + 1;
    return {
      attempt,
      path:
        `${directory}/${stem}.attempt-` +
        `${String(attempt).padStart(2, "0")}.raw.txt`,
    };
  };

  const provenanceFor = (active) => {
    const teacherMode = active?.teacher_mode ?? null;
    return {
      teacher_mode: teacherMode,
      counts_toward_pro_token_target:
        typeof teacherMode === "string" && /Pro/.test(teacherMode),
    };
  };

  const saveResponse = async (job) => {
    const active = await readJson(fs, activePath);
    const raw = await readLatestResponse();
    const { attempt, path: attemptPath } = await nextAttemptPath(
      rawDir,
      job.web_job_id,
    );
    await fs.writeFile(attemptPath, raw, { encoding: "utf8", flag: "wx" });
    await fs.writeFile(
      `${attemptPath}.provenance.json`,
      `${JSON.stringify(provenanceFor(active), null, 2)}\n`,
      { encoding: "utf8", flag: "wx" },
    );
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch {}
    const status = parsed ? "raw-saved" : "invalid-response";
    await writeState({
      status,
      web_job_id: job.web_job_id,
      attempt,
      raw_path: attemptPath,
      teacher_mode: active?.teacher_mode ?? null,
      recorded_at: new Date().toISOString(),
    });
    return {
      job: job.web_job_id,
      status,
      attempt,
      characters: raw.length,
      rawPreview: parsed ? undefined : raw.slice(0, 120),
    };
  };

  const saveGroupedResponse = async (group, members) => {
    const active = await readJson(fs, activePath);
    const raw = await readLatestResponse();
    const { attempt, path: groupPath } = await nextAttemptPath(
      rawGroupDir,
      group.web_job_id,
    );
    await fs.writeFile(groupPath, raw, { encoding: "utf8", flag: "wx" });
    await fs.writeFile(
      `${groupPath}.provenance.json`,
      `${JSON.stringify(provenanceFor(active), null, 2)}\n`,
      { encoding: "utf8", flag: "wx" },
    );
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch {}
    let splitStatus = "invalid-response";
    const splitPaths = [];
    if (Array.isArray(parsed)) {
      const expectedIds = group.expected_conversations.map(
        (row) => row.conversation_id,
      );
      const observedIds = parsed.map((row) => row?.conversation_id);
      const observedCounts = new Map();
      for (const conversationId of observedIds) {
        observedCounts.set(
          conversationId,
          (observedCounts.get(conversationId) ?? 0) + 1,
        );
      }
      const exactSet =
        observedIds.length === expectedIds.length &&
        expectedIds.every((conversationId) =>
          observedCounts.get(conversationId) === 1
        );
      if (exactSet) {
        const byConversationId = new Map(
          parsed.map((row) => [row.conversation_id, row]),
        );
        for (const member of members) {
          const memberRows = member.expected_conversations.map((row) =>
            byConversationId.get(row.conversation_id)
          );
          const memberAttempt = await nextAttemptPath(
            rawDir,
            member.web_job_id,
          );
          await fs.writeFile(
            memberAttempt.path,
            `${JSON.stringify(memberRows, null, 2)}\n`,
            { encoding: "utf8", flag: "wx" },
          );
          await fs.writeFile(
            `${memberAttempt.path}.provenance.json`,
            `${JSON.stringify(provenanceFor(active), null, 2)}\n`,
            { encoding: "utf8", flag: "wx" },
          );
          splitPaths.push(memberAttempt.path);
        }
        const manifest = {
          group_job_id: group.web_job_id,
          source_path: groupPath,
          teacher_mode: active?.teacher_mode ?? null,
          source_content_changed: false,
          split_operation: "select conversations by exact conversation_id",
          member_paths: splitPaths,
        };
        await fs.writeFile(
          `${groupPath}.split-manifest.json`,
          `${JSON.stringify(manifest, null, 2)}\n`,
          { encoding: "utf8", flag: "wx" },
        );
        splitStatus = "raw-saved-group";
      }
    }
    await writeState({
      status: splitStatus,
      web_job_id: group.web_job_id,
      member_job_ids: members.map((member) => member.web_job_id),
      attempt,
      raw_path: groupPath,
      split_paths: splitPaths,
      teacher_mode: active?.teacher_mode ?? null,
      recorded_at: new Date().toISOString(),
    });
    return {
      job: group.web_job_id,
      status: splitStatus,
      attempt,
      characters: raw.length,
      splitJobs: splitPaths.length,
      rawPreview: parsed ? undefined : raw.slice(0, 120),
    };
  };

  const collect = async (job, pollSeconds = 200, saver = saveResponse) => {
    const polls = Math.max(1, Math.floor(pollSeconds / 5));
    let previousRaw = "";
    let stablePolls = 0;
    let revertedPolls = 0;
    for (let index = 0; index < polls; index += 1) {
      const stopButton = tab.playwright.getByRole("button", {
        name: "Stop response",
      });
      const stop = await stopButton.count();
      const raw = await readLatestResponse();
      if (stop === 0 && raw.trim()) return saver(job);
      if (stop === 0 && !raw.trim()) {
        const sentCount = await tab.playwright.locator("user-query").count();
        const box = tab.playwright.getByRole("textbox", {
          name: "Enter a prompt for Gemini",
        });
        const draft = (await box.count()) ? await box.innerText() : "";
        const hasExpectedDraft = job.expected_conversations.every((row) =>
          draft.includes(row.conversation_id)
        );
        revertedPolls = sentCount === 0 && hasExpectedDraft
          ? revertedPolls + 1
          : 0;
        if (revertedPolls >= 2) {
          await writeState({
            status: "submit-reverted",
            web_job_id: job.web_job_id,
            recorded_at: new Date().toISOString(),
          });
          return { job: job.web_job_id, status: "submit-reverted" };
        }
      }
      if (raw.trim()) {
        stablePolls = raw === previousRaw ? stablePolls + 1 : 0;
        previousRaw = raw;
        let parsed = null;
        try {
          parsed = JSON.parse(raw);
        } catch {}
        const expectedIds = new Set(
          job.expected_conversations.map((row) => row.conversation_id),
        );
        const observedIds = Array.isArray(parsed)
          ? parsed.map((row) => row?.conversation_id)
          : [];
        const complete =
          observedIds.length === expectedIds.size &&
          observedIds.every((conversationId) => expectedIds.has(conversationId)) &&
          new Set(observedIds).size === observedIds.length;
        if (complete && stablePolls >= 2) {
          if (stop) {
            await stopButton.click();
            await sleep(tab, 600);
          }
          return saver(job);
        }
      }
      await sleep(tab, 5000);
    }
    return { job: job.web_job_id, status: "still-active" };
  };

  const buildGroup = (members) => {
    const marker = "Batch specification:\n\n";
    const markerIndex = members[0].prompt.indexOf(marker);
    if (markerIndex < 0) throw new Error("batch specification marker missing");
    const expectedConversations = members.flatMap(
      (member) => member.expected_conversations,
    );
    const first = members[0].web_job_id.slice(-4);
    const last = members.at(-1).web_job_id.slice(-4);
    return {
      web_job_id: `KINYA-GEMINI-WEB-GROUP-${first}-${last}`,
      expected_conversations: expectedConversations,
      prompt:
        members[0].prompt.slice(0, markerIndex + marker.length) +
        `${JSON.stringify(expectedConversations, null, 2)}\n`,
    };
  };

  const runGroup = async (jobIds, pollSeconds = 200) => {
    if (!Array.isArray(jobIds) || jobIds.length !== 2) {
      throw new Error("runGroup requires exactly two web job IDs");
    }
    const members = jobIds.map((jobId) => {
      const job = byId.get(jobId);
      if (!job) throw new Error(`unknown web job: ${jobId}`);
      return job;
    });
    for (const member of members) {
      try {
        await fs.access(
          `${validatedDir}/${member.web_job_id}.validated.json`,
        );
        throw new Error(`already validated: ${member.web_job_id}`);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
    const group = buildGroup(members);
    const active = await readJson(fs, activePath);
    if (active?.status === "active") {
      if (active.web_job_id !== group.web_job_id) {
        throw new Error(`another web job is active: ${active.web_job_id}`);
      }
      const answerNow = tab.playwright.getByRole("button", {
        name: "Answer now",
      });
      if (await answerNow.count()) await answerNow.click();
    } else {
      await submit(group);
    }
    return collect(
      group,
      pollSeconds,
      () => saveGroupedResponse(group, members),
    );
  };

  const run = async (jobId, pollSeconds = 200) => {
    const job = byId.get(jobId);
    if (!job) throw new Error(`unknown web job: ${jobId}`);
    try {
      await fs.access(`${validatedDir}/${jobId}.validated.json`);
      return { job: jobId, status: "already-validated" };
    } catch {}
    const active = await readJson(fs, activePath);
    if (active?.status === "active") {
      if (active.web_job_id !== jobId) {
        throw new Error(`another web job is active: ${active.web_job_id}`);
      }
      const answerNow = tab.playwright.getByRole("button", {
        name: "Answer now",
      });
      if (await answerNow.count()) await answerNow.click();
    } else {
      await submit(job);
    }
    return collect(job, pollSeconds);
  };

  return { jobs, run, runGroup };
}
