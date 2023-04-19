import https from 'https';
import type { ClientRequest, IncomingMessage } from 'http';
import type { ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionResponse } from 'openai';
import { type TiktokenModel } from '@dqbd/tiktoken';
import createHttpsProxyAgent from 'https-proxy-agent';
import { KnownError } from './error.js';
import type { CommitType } from './config.js';

const httpsPost = async (
	hostname: string,
	path: string,
	headers: Record<string, string>,
	json: unknown,
	timeout: number,
	proxy?: string,
) => new Promise<{
	request: ClientRequest;
	response: IncomingMessage;
	data: string;
}>((resolve, reject) => {
	const postContent = JSON.stringify(json);
	const request = https.request(
		{
			port: 443,
			hostname,
			path,
			method: 'POST',
			headers: {
				...headers,
				'Content-Type': 'application/json',
				'Content-Length': Buffer.byteLength(postContent),
			},
			timeout,
			agent: (
				proxy
					? createHttpsProxyAgent(proxy)
					: undefined
			),
		},
		(response) => {
			const body: Buffer[] = [];
			response.on('data', chunk => body.push(chunk));
			response.on('end', () => {
				resolve({
					request,
					response,
					data: Buffer.concat(body).toString(),
				});
			});
		},
	);
	request.on('error', reject);
	request.on('timeout', () => {
		request.destroy();
		reject(new KnownError(`Time out error: request took over ${timeout}ms. Try increasing the \`timeout\` config, or checking the OpenAI API status https://status.openai.com`));
	});

	request.write(postContent);
	request.end();
});

const createChatCompletion = async (
	apiKey: string,
	json: CreateChatCompletionRequest,
	timeout: number,
	proxy?: string,
) => {
	const { response, data } = await httpsPost(
		'api.openai.com',
		'/v1/chat/completions',
		{
			Authorization: `Bearer ${apiKey}`,
		},
		json,
		timeout,
		proxy,
	);

	if (
		!response.statusCode
		|| response.statusCode < 200
		|| response.statusCode > 299
	) {
		let errorMessage = `OpenAI API Error: ${response.statusCode} - ${response.statusMessage}`;

		if (data) {
			errorMessage += `\n\n${data}`;
		}

		if (response.statusCode === 500) {
			errorMessage += '\n\nCheck the API status: https://status.openai.com';
		}

		throw new KnownError(errorMessage);
	}

	return JSON.parse(data) as CreateChatCompletionResponse;
};

const sanitizeMessage = (message: string) => message.trim().replace(/[\n\r]/g, '').replace(/(\w)\.$/, '$1');

const deduplicateMessages = (array: string[]) => Array.from(new Set(array));

const getBasePrompt = (locale: string) => `Write good commit messages in the language ${locale} that follow the following 7 rules.
Rules:
1. Separate subject from body with a blank line.
2. Limit the subject line to 72 characters.
3. Captialize the subject line.
4. Do not end the subject line with a period.
5. Use the imperative mood in the subject line.
6. Wrap the body at 80 characters.
7. Use the body to explain what and why vs. how.

For example:

    Summarize changes in around 50 characters or less

    More detailed explanatory text, if necessary. Wrap it to about 72
    characters or so. In some contexts, the first line is treated as the
    subject of the commit and the rest of the text as the body. The
    blank line separating the summary from the body is critical (unless
    you omit the body entirely); various tools like \`log\`, \`shortlog\`
    and \`rebase\` can get confused if you run the two together.

    Explain the problem that this commit is solving. Focus on why you
    are making this change as opposed to how (the code explains that).
    Are there side effects or other unintuitive consequences of this
    change? Here's the place to explain them.

    Further paragraphs come after blank lines.

     - Bullet points are okay, too

     - Typically a hyphen or asterisk is used for the bullet, preceded
       by a single space, with blank lines in between, but conventions
       vary here
`;

const getCommitMessageFormatPrompt = (type: CommitType) => {
	if (type === 'conventional') {
		return '<type>(<optional scope>): <commit message>';
	}

	return '<commit message>';
};

const getExtraContextForConventionalCommits = () => {
	/**
	 * References:
	 * Commitlint:
	 * https://github.com/conventional-changelog/commitlint/blob/18fbed7ea86ac0ec9d5449b4979b762ec4305a92/%40commitlint/config-conventional/index.js#L40-L100
	 *
	 * Conventional Changelog:
	 * https://github.com/conventional-changelog/conventional-changelog/blob/d0e5d5926c8addba74bc962553dd8bcfba90e228/packages/conventional-changelog-conventionalcommits/writer-opts.js#L182-L193
	 */
	const conventionalCommitTypes: Record<string, string> = {
		/*
			Commented out because they are too common and
			will cause the model to generate them too often.
		*/
		docs: 'Documentation only changes',
		style:
			'Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)',
		refactor: 'A code change that neither fixes a bug nor adds a feature',
		perf: 'A code change that improves performance',
		test: 'Adding missing tests or correcting existing tests',
		build: 'Changes that affect the build system or external dependencies',
		ci: 'Changes to our CI configuration files and scripts',
		chore: "Other changes that don't modify src or test files",
		revert: 'Reverts a previous commit',
		feat: 'A new feature',
		fix: 'A bug fix',
	};

	return `Choose a type from the type-to-description JSON below that best describes the git diff:\n${JSON.stringify(conventionalCommitTypes, null, 2)}`;
};

export const generateCommitMessage = async (
	apiKey: string,
	model: TiktokenModel,
	locale: string,
	diff: string,
	completions: number,
	type: CommitType,
	timeout: number,
	proxy?: string,
) => {
	const basePrompt = getBasePrompt(locale);

	const commitMessageFormatPrompt = getCommitMessageFormatPrompt(
		type,
	);

	const conventionalCommitsExtraContext = type === 'conventional'
		? getExtraContextForConventionalCommits()
		: '';

	const messages: ChatCompletionRequestMessage[] = [
		{
			role: 'system',
			content: `${basePrompt}\n${commitMessageFormatPrompt}\n${conventionalCommitsExtraContext}`,
		},
		{
			role: 'user',
			content: diff,
		},
	];

	try {
		const completion = await createChatCompletion(apiKey, {
			model,
			messages,
			temperature: 0.7,
			top_p: 1,
			frequency_penalty: 0,
			presence_penalty: 0,
			max_tokens: 200,
			stream: false,
			n: completions,
		}, timeout, proxy);

		return deduplicateMessages(
			completion.choices
				.filter(choice => choice.message?.content)
				.map(choice => choice.message!.content),
		);
	} catch (error) {
		const errorAsAny = error as any;
		if (errorAsAny.code === 'ENOTFOUND') {
			throw new KnownError(`Error connecting to ${errorAsAny.hostname} (${errorAsAny.syscall}). Are you connected to the internet?`);
		}

		throw errorAsAny;
	}
};
